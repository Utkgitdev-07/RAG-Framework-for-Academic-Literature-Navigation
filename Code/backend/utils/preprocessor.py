"""
Text Preprocessing Utilities Module

This module provides text cleaning, normalization, and preparation functions
for embedding generation. It handles text chunking, stopword removal (optional),
and formatting for optimal embedding quality.

Key Features:
- Text cleaning (removal of URLs, emails, special characters)
- Text chunking with overlapping windows
- Sentence extraction for summarization
- Text normalization and whitespace handling
- Metadata and text combination for enriched embeddings

Processing Steps:
1. Clean text (remove URLs, normalize whitespace)
2. Optionally remove stopwords
3. Chunk long documents into overlapping segments
4. Prepare text for embedding (truncate to max length)
5. Combine metadata with text for richer context

Author: Research Team
Date: 2024
"""
import re
from typing import List
import nltk

# Download required NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


class TextPreprocessor:
    """Preprocesses text for embedding and retrieval"""
    
    def __init__(self, remove_stopwords=False):
        self.remove_stopwords = remove_stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Size of each chunk (characters)
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending near chunk boundary
                chunk_text = text[start:end]
                last_period = chunk_text.rfind('.')
                last_exclamation = chunk_text.rfind('!')
                last_question = chunk_text.rfind('?')
                
                last_sentence = max(last_period, last_exclamation, last_question)
                
                if last_sentence > chunk_size * 0.7:  # If sentence ending is in last 30%
                    end = start + last_sentence + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def extract_key_sentences(self, text: str, num_sentences: int = 5) -> List[str]:
        """
        Extract key sentences from text (simplified - first N sentences)
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            
        Returns:
            List of key sentences
        """
        sentences = sent_tokenize(text)
        # Return first N sentences (can be improved with TF-IDF or other methods)
        return sentences[:num_sentences]
    
    def prepare_for_embedding(self, text: str, max_length: int = 512) -> str:
        """
        Prepare text for embedding (clean and truncate)
        
        Args:
            text: Input text
            max_length: Maximum length
            
        Returns:
            Prepared text
        """
        text = self.clean_text(text)
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]  # Break at word boundary
        
        return text
    
    def combine_text_metadata(self, text: str, metadata: dict) -> str:
        """
        Combine text with metadata for richer embedding
        
        Args:
            text: Main text
            metadata: Metadata dictionary
            
        Returns:
            Combined text string
        """
        parts = []
        
        # Add title if available
        if metadata.get('title'):
            parts.append(f"Title: {metadata['title']}")
        
        # Add abstract
        if metadata.get('abstract'):
            parts.append(f"Abstract: {metadata['abstract']}")
        
        # Add keywords
        if metadata.get('keywords'):
            keywords_str = ', '.join(metadata['keywords'][:5])
            parts.append(f"Keywords: {keywords_str}")
        
        # Add main text
        parts.append(text)
        
        return ' '.join(parts)

