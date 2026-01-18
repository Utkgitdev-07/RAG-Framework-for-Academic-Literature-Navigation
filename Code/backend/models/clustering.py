"""
Topic Clustering Module for RAG Literature Navigation System

This module implements K-means clustering for organizing search results into
topical groups. It includes automatic optimal cluster count selection using
silhouette score evaluation.

Key Features:
- K-means clustering on document embeddings
- Optimal K selection using silhouette score
- Cluster summary generation
- Keyword extraction from clusters

Author: Research Team
Date: 2024
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Optional
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)


class TopicClustering:
    """Performs topic clustering on search results"""
    
    def __init__(self, embedder):
        """
        Initialize clustering system
        
        Args:
            embedder: EmbeddingGenerator instance
        """
        self.embedder = embedder
        logger.info("Topic clustering system initialized")
    
    def cluster_documents(
        self,
        documents: List[Dict],
        num_clusters: Optional[int] = None,
        min_clusters: int = 3,
        max_clusters: int = 10
    ) -> Dict:
        """
        Cluster documents into topics
        
        Args:
            documents: List of documents with 'embedding' field or 'text' field
            num_clusters: Fixed number of clusters (None for auto-selection)
            min_clusters: Minimum number of clusters for auto-selection
            max_clusters: Maximum number of clusters for auto-selection
            
        Returns:
            Dictionary with cluster assignments and statistics
        """
        if len(documents) < min_clusters:
            logger.warning(f"Not enough documents for clustering (need at least {min_clusters})")
            return {
                'clusters': [],
                'labels': [],
                'num_clusters': 0,
                'silhouette_score': 0.0
            }
        
        # Extract embeddings
        embeddings = []
        for doc in documents:
            if 'embedding' in doc:
                emb = doc['embedding']
            elif 'text' in doc:
                emb = self.embedder.encode_text(doc['text'])
            else:
                raise ValueError("Document must have 'embedding' or 'text' field")
            
            if isinstance(emb, np.ndarray) and len(emb.shape) == 1:
                embeddings.append(emb)
            else:
                embeddings.append(emb[0] if len(emb.shape) > 1 else emb)
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        # Determine optimal K if not specified
        if num_clusters is None:
            num_clusters = self._select_optimal_k(
                embeddings,
                min_clusters,
                min(max_clusters, len(documents))
            )
        
        # Perform clustering
        logger.info(f"Clustering {len(documents)} documents into {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
            silhouette = silhouette_score(embeddings, labels)
        else:
            silhouette = 0.0
        
        # Organize documents by cluster
        clusters = {}
        for i, (doc, label) in enumerate(zip(documents, labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doc)
        
        return {
            'clusters': clusters,
            'labels': labels.tolist(),
            'num_clusters': num_clusters,
            'silhouette_score': float(silhouette),
            'centroids': kmeans.cluster_centers_.tolist()
        }
    
    def _select_optimal_k(self, embeddings: np.ndarray, min_k: int, max_k: int) -> int:
        """
        Select optimal number of clusters using silhouette score
        
        Args:
            embeddings: Document embeddings
            min_k: Minimum number of clusters
            max_k: Maximum number of clusters
            
        Returns:
            Optimal number of clusters
        """
        if max_k - min_k < 1:
            return min_k
        
        best_k = min_k
        best_score = -1
        
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception as e:
                logger.warning(f"Error clustering with k={k}: {e}")
                continue
        
        logger.info(f"Selected optimal k={best_k} (silhouette score: {best_score:.3f})")
        return best_k
    
    def format_clustered_results(self, clustering_result: Dict) -> List[Dict]:
        """
        Format clustering results with summaries and keywords
        
        Args:
            clustering_result: Result from cluster_documents()
            
        Returns:
            List of formatted cluster dictionaries
        """
        clusters = clustering_result.get('clusters', {})
        formatted_clusters = []
        
        for cluster_id, docs in clusters.items():
            if not docs:
                continue
            
            # Extract keywords from cluster documents
            keywords = self._extract_cluster_keywords(docs)
            
            # Generate summary
            summary = self._generate_cluster_summary(cluster_id, docs, keywords)
            
            # Sort documents by score if available
            sorted_docs = sorted(
                docs,
                key=lambda d: d.get('score', 0.0),
                reverse=True
            )
            
            formatted_cluster = {
                'cluster_id': int(cluster_id),
                'num_documents': len(docs),
                'summary': summary,
                'keywords': keywords[:10],  # Top 10 keywords
                'documents': sorted_docs
            }
            
            formatted_clusters.append(formatted_cluster)
        
        # Sort clusters by number of documents (descending)
        formatted_clusters.sort(key=lambda c: c['num_documents'], reverse=True)
        
        return formatted_clusters
    
    def _extract_cluster_keywords(self, documents: List[Dict], top_n: int = 10) -> List[str]:
        """
        Extract common keywords from cluster documents
        
        Args:
            documents: Documents in the cluster
            top_n: Number of keywords to return
            
        Returns:
            List of top keywords
        """
        # Collect all keywords from metadata
        all_keywords = []
        all_text = []
        
        for doc in documents:
            # Get keywords from metadata
            if 'metadata' in doc and doc['metadata']:
                meta = doc['metadata']
                if 'keywords' in meta and meta['keywords']:
                    all_keywords.extend([k.lower() for k in meta['keywords']])
            
            # Extract words from text (simple approach)
            if 'text' in doc:
                text = doc['text'][:500]  # Use first 500 chars
                words = re.findall(r'\b[a-z]{4,}\b', text.lower())
                all_text.extend(words)
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        
        # If not enough keywords, use common words from text
        if len(keyword_counts) < top_n:
            text_counts = Counter(all_text)
            # Filter out common stopwords
            stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their', 'there', 'these', 'would', 'could', 'should'}
            filtered_text = {k: v for k, v in text_counts.items() if k not in stopwords}
            keyword_counts.update(filtered_text)
        
        # Get top keywords
        top_keywords = [word for word, count in keyword_counts.most_common(top_n)]
        
        return top_keywords
    
    def _generate_cluster_summary(self, cluster_id: int, documents: List[Dict], keywords: List[str]) -> str:
        """
        Generate a summary for a cluster
        
        Args:
            cluster_id: Cluster identifier
            documents: Documents in the cluster
            keywords: Top keywords for the cluster
            
        Returns:
            Summary string
        """
        num_docs = len(documents)
        
        # Get common topics from titles if available
        titles = []
        for doc in documents[:5]:  # Check first 5 documents
            if 'metadata' in doc and doc['metadata']:
                title = doc['metadata'].get('title', '')
                if title:
                    titles.append(title[:50])  # Truncate long titles
        
        # Build summary
        parts = [f"Cluster {cluster_id + 1} contains {num_docs} document"]
        if num_docs > 1:
            parts[0] += "s"
        
        if keywords:
            keywords_str = ', '.join(keywords[:5])
            parts.append(f"Common topics: {keywords_str}.")
        
        if titles:
            parts.append(f"Sample titles: {titles[0]}.")
        
        return ' '.join(parts)
