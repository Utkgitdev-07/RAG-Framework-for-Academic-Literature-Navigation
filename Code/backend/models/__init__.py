"""
Models package for RAG Literature Navigation System

This package contains the core model components:
- EmbeddingGenerator: Generates embeddings for text and metadata
- RetrievalSystem: Manages FAISS indices and hybrid search
- TopicClustering: Performs K-means clustering on search results
"""

from .embedding import EmbeddingGenerator
from .retrieval import RetrievalSystem
from .clustering import TopicClustering

__all__ = ['EmbeddingGenerator', 'RetrievalSystem', 'TopicClustering']
