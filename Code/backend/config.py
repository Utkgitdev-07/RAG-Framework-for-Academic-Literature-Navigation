"""
Configuration Module for RAG Literature Navigation System

This module contains all system configuration parameters including file paths,
model settings, retrieval parameters, clustering options, and API configuration.
All tunable parameters are centralized here for easy customization.

Configuration Categories:
- File Paths: Data directories and index storage locations
- Embedding Model: Model selection and dimension settings
- Retrieval: Top-K results, hybrid scoring weights
- Clustering: Cluster count limits and optimization parameters
- Text Processing: Chunk sizes, text length limits
- API: Server host, port, debug settings
- PDF Processing: File size limits and supported formats

Key Parameters:
- TEXT_WEIGHT: 0.7 (70% importance for text content in hybrid search)
- METADATA_WEIGHT: 0.3 (30% importance for metadata in hybrid search)
- EMBEDDING_DIM: 384 (dimension of sentence transformer embeddings)

Author: Research Team
Date: 2024
"""
import os

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data" / "papers"
INDEX_DIR = BACKEND_DIR / "data" / "index"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Embedding Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval Configuration
TOP_K = 10  # Default number of results
MAX_RESULTS = 50

# Clustering Configuration
NUM_CLUSTERS = 5
MIN_CLUSTERS = 3
MAX_CLUSTERS = 10

# Retrieval Weights (for hybrid scoring)
TEXT_WEIGHT = 0.7  # Weight for text similarity
METADATA_WEIGHT = 0.3  # Weight for metadata similarity

# Text Processing
MAX_TEXT_LENGTH = 512
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
DEBUG = True

# PDF Processing
SUPPORTED_EXTENSIONS = ['.pdf']
MAX_PDF_SIZE_MB = 50

