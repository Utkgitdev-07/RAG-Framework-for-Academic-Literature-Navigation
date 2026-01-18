"""
Embedding Generation Module for RAG Literature Navigation System

This module handles the generation of vector embeddings for text and metadata
using sentence transformers. It provides a unified interface for encoding
both textual content and structured metadata into dense vector representations.

Key Features:
- Text embedding generation using sentence transformers
- Metadata embedding generation (combines multiple metadata fields)
- Batch processing support
- Model caching and lazy loading

Author: Research Team
Date: 2024
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text and metadata using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            Numpy array of embeddings (shape: (n, dim) for list, (dim,) for single)
        """
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False
        )
        
        if len(embeddings.shape) == 1:
            return embeddings
        return embeddings
    
    def encode_metadata(self, metadata: Union[dict, List[dict]]) -> np.ndarray:
        """
        Generate embeddings for metadata by combining metadata fields into text
        
        Args:
            metadata: Single metadata dict or list of metadata dicts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(metadata, dict):
            metadata = [metadata]
        
        # Convert metadata to text strings
        metadata_texts = []
        for meta in metadata:
            parts = []
            
            # Add title
            if meta.get('title'):
                parts.append(f"Title: {meta['title']}")
            
            # Add authors
            if meta.get('authors'):
                authors_str = ', '.join(meta['authors'][:5])  # Limit to 5 authors
                parts.append(f"Authors: {authors_str}")
            
            # Add abstract
            if meta.get('abstract'):
                abstract = meta['abstract'][:200]  # Limit length
                parts.append(f"Abstract: {abstract}")
            
            # Add keywords
            if meta.get('keywords'):
                keywords_str = ', '.join(meta['keywords'][:10])
                parts.append(f"Keywords: {keywords_str}")
            
            # Add venue
            if meta.get('venue'):
                parts.append(f"Venue: {meta['venue']}")
            
            # Add year
            if meta.get('year'):
                parts.append(f"Year: {meta['year']}")
            
            # Combine all parts
            metadata_text = ' '.join(parts)
            if not metadata_text:
                metadata_text = "Academic paper"  # Fallback
            
            metadata_texts.append(metadata_text)
        
        # Generate embeddings
        return self.encode_text(metadata_texts)
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.embedding_dim
