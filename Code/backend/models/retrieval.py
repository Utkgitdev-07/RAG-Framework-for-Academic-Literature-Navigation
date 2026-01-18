"""
Retrieval System Module for RAG Literature Navigation System

This module implements the hybrid retrieval system using FAISS for efficient
similarity search. It supports both text-only and multi-modal (text + metadata)
retrieval with weighted score fusion.

Key Features:
- FAISS-based vector storage and search
- Hybrid retrieval combining text and metadata scores
- Index persistence (save/load from disk)
- Weighted score fusion (configurable weights)

Author: Research Team
Date: 2024
"""
import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from config import INDEX_DIR, TEXT_WEIGHT, METADATA_WEIGHT, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class RetrievalSystem:
    """Manages document retrieval using FAISS indices"""
    
    def __init__(self, embedder):
        """
        Initialize retrieval system
        
        Args:
            embedder: EmbeddingGenerator instance
        """
        self.embedder = embedder
        self.embedding_dim = embedder.get_embedding_dim()
        
        # Document storage
        self.documents: List[Dict] = []
        self.metadata_list: List[Dict] = []
        
        # FAISS indices
        self.text_index: Optional[faiss.Index] = None
        self.metadata_index: Optional[faiss.Index] = None
        
        # Index state
        self.is_indexed = False
        
        # Weights for hybrid search
        self.text_weight = TEXT_WEIGHT
        self.metadata_weight = METADATA_WEIGHT
        
        logger.info(f"Retrieval system initialized. Embedding dim: {self.embedding_dim}")
    
    def add_documents(self, documents: List[Dict], metadata_list: List[Dict]):
        """
        Add documents to the retrieval system
        
        Args:
            documents: List of document dictionaries with 'text' field
            metadata_list: List of metadata dictionaries
        """
        if len(documents) != len(metadata_list):
            raise ValueError("Documents and metadata lists must have same length")
        
        self.documents.extend(documents)
        self.metadata_list.extend(metadata_list)
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def build_index(self):
        """Build FAISS indices from added documents"""
        if not self.documents:
            raise ValueError("No documents to index")
        
        num_docs = len(self.documents)
        logger.info(f"Building indices for {num_docs} documents...")
        
        # Generate embeddings
        texts = [doc['text'] for doc in self.documents]
        text_embeddings = self.embedder.encode_text(texts)
        
        # Generate metadata embeddings
        metadata_embeddings = self.embedder.encode_metadata(self.metadata_list)
        
        # Normalize embeddings (L2 normalization for cosine similarity)
        faiss.normalize_L2(text_embeddings)
        faiss.normalize_L2(metadata_embeddings)
        
        # Create FAISS indices (Flat index for exact search)
        self.text_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product = Cosine for normalized vectors
        self.metadata_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to indices
        self.text_index.add(text_embeddings.astype('float32'))
        self.metadata_index.add(metadata_embeddings.astype('float32'))
        
        self.is_indexed = True
        logger.info(f"Indices built successfully. Text index: {self.text_index.ntotal}, Metadata index: {self.metadata_index.ntotal}")
    
    def search(self, query: str, top_k: int = 10, use_hybrid: bool = True) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query string
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid (text + metadata) search
            
        Returns:
            List of ranked documents with scores
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedder.encode_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        # Search text index (always)
        search_k = top_k * 2 if use_hybrid else top_k  # Get more candidates for fusion
        text_scores, text_indices = self.text_index.search(query_embedding, min(search_k, len(self.documents)))
        text_scores = text_scores[0]
        text_indices = text_indices[0]
        
        if use_hybrid:
            # Search metadata index
            metadata_scores, metadata_indices = self.metadata_index.search(query_embedding, min(search_k, len(self.documents)))
            metadata_scores = metadata_scores[0]
            metadata_indices = metadata_indices[0]
            
            # Combine scores using weighted fusion
            candidate_scores = {}
            
            # Process text results
            for i, (score, idx) in enumerate(zip(text_scores, text_indices)):
                if idx < 0:  # Invalid index
                    continue
                doc_id = idx
                if doc_id not in candidate_scores:
                    candidate_scores[doc_id] = {
                        'text_score': float(score),
                        'metadata_score': 0.0,
                        'doc_id': doc_id
                    }
                else:
                    candidate_scores[doc_id]['text_score'] = max(candidate_scores[doc_id]['text_score'], float(score))
            
            # Process metadata results
            for i, (score, idx) in enumerate(zip(metadata_scores, metadata_indices)):
                if idx < 0:
                    continue
                doc_id = idx
                if doc_id not in candidate_scores:
                    candidate_scores[doc_id] = {
                        'text_score': 0.0,
                        'metadata_score': float(score),
                        'doc_id': doc_id
                    }
                else:
                    candidate_scores[doc_id]['metadata_score'] = max(candidate_scores[doc_id]['metadata_score'], float(score))
            
            # Calculate combined scores
            for doc_id, scores in candidate_scores.items():
                combined = (scores['text_score'] * self.text_weight) + (scores['metadata_score'] * self.metadata_weight)
                scores['score'] = combined
            
            # Sort by combined score
            ranked_candidates = sorted(candidate_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # Build results
            results = []
            for rank, (doc_id, scores) in enumerate(ranked_candidates[:top_k], 1):
                doc = self.documents[doc_id].copy()
                doc['score'] = scores['score']
                doc['text_score'] = scores['text_score']
                doc['metadata_score'] = scores['metadata_score']
                doc['rank'] = rank
                doc['metadata'] = self.metadata_list[doc_id]
                results.append(doc)
        else:
            # Text-only search
            results = []
            for rank, (score, idx) in enumerate(zip(text_scores, text_indices), 1):
                if idx < 0:
                    continue
                if len(results) >= top_k:
                    break
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                doc['text_score'] = float(score)
                doc['rank'] = rank
                doc['metadata'] = self.metadata_list[idx]
                results.append(doc)
        
        return results
    
    def save_index(self):
        """Save indices and documents to disk"""
        if not self.is_indexed:
            logger.warning("No index to save")
            return
        
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS indices
        if self.text_index:
            faiss.write_index(self.text_index, str(INDEX_DIR / "text_index.faiss"))
        if self.metadata_index:
            faiss.write_index(self.metadata_index, str(INDEX_DIR / "metadata_index.faiss"))
        
        # Save documents and metadata
        with open(INDEX_DIR / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        with open(INDEX_DIR / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Indices saved to {INDEX_DIR}")
    
    def load_index(self):
        """Load indices and documents from disk"""
        text_index_path = INDEX_DIR / "text_index.faiss"
        metadata_index_path = INDEX_DIR / "metadata_index.faiss"
        documents_path = INDEX_DIR / "documents.json"
        metadata_path = INDEX_DIR / "metadata.json"
        
        if not all(p.exists() for p in [text_index_path, metadata_index_path, documents_path, metadata_path]):
            logger.info("No saved indices found. Starting fresh.")
            return
        
        try:
            # Load FAISS indices
            self.text_index = faiss.read_index(str(text_index_path))
            self.metadata_index = faiss.read_index(str(metadata_index_path))
            
            # Load documents and metadata
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata_list = json.load(f)
            
            self.is_indexed = True
            logger.info(f"Loaded indices: {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            self.is_indexed = False
    
    def get_stats(self) -> Dict:
        """Get statistics about the index"""
        return {
            'indexed': self.is_indexed,
            'num_documents': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'text_index_size': self.text_index.ntotal if self.text_index else 0,
            'metadata_index_size': self.metadata_index.ntotal if self.metadata_index else 0
        }
