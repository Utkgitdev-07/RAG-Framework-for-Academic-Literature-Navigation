"""
PDF Parser Module for Academic Paper Text Extraction

This module handles parsing of PDF documents to extract textual content and
basic metadata. It uses multiple parsing strategies (pdfplumber and PyPDF2)
to ensure robust text extraction from various PDF formats.

Key Features:
- Primary parser: pdfplumber for better text extraction quality
- Fallback parser: PyPDF2 for compatibility
- Extracts text from all pages
- Captures PDF metadata (title, author, creation date)
- Abstract section detection and extraction

Technical Approach:
- Iterates through all pages in the PDF
- Extracts raw text content
- Captures document metadata from PDF properties
- Handles extraction errors gracefully

Author: Research Team
Date: 2024
"""
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFParser:
    """Extracts text and metadata from PDF files"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def parse(self, pdf_path: str) -> Dict:
        """
        Parse a PDF file and extract text and metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with text, metadata, and pages
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {pdf_path.suffix}")
        
        result = {
            'file_path': str(pdf_path),
            'file_name': pdf_path.name,
            'text': '',
            'pages': [],
            'metadata': {},
            'error': None
        }
        
        try:
            # Try pdfplumber first (better for text extraction)
            result.update(self._parse_with_pdfplumber(pdf_path))
        except Exception as e1:
            logger.warning(f"pdfplumber failed for {pdf_path.name}, trying PyPDF2: {e1}")
            try:
                result.update(self._parse_with_pypdf2(pdf_path))
            except Exception as e2:
                logger.error(f"Both parsers failed for {pdf_path.name}: {e2}")
                result['error'] = str(e2)
        
        return result
    
    def _parse_with_pdfplumber(self, pdf_path: Path) -> Dict:
        """Parse PDF using pdfplumber (better text extraction)"""
        text_parts = []
        pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata or {}
            
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                pages.append({
                    'page_num': i,
                    'text': page_text,
                    'bbox': page.bbox if hasattr(page, 'bbox') else None
                })
        
        full_text = '\n\n'.join(text_parts)
        
        return {
            'text': full_text,
            'pages': pages,
            'metadata': {
                'title': metadata.get('Title', ''),
                'author': metadata.get('Author', ''),
                'subject': metadata.get('Subject', ''),
                'creator': metadata.get('Creator', ''),
                'producer': metadata.get('Producer', ''),
                'creation_date': metadata.get('CreationDate', ''),
                'modification_date': metadata.get('ModDate', ''),
                'num_pages': len(pages)
            }
        }
    
    def _parse_with_pypdf2(self, pdf_path: Path) -> Dict:
        """Fallback parser using PyPDF2"""
        text_parts = []
        pages = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = pdf_reader.metadata or {}
            
            for i, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                pages.append({
                    'page_num': i,
                    'text': page_text
                })
        
        full_text = '\n\n'.join(text_parts)
        
        return {
            'text': full_text,
            'pages': pages,
            'metadata': {
                'title': metadata.get('/Title', ''),
                'author': metadata.get('/Author', ''),
                'subject': metadata.get('/Subject', ''),
                'creator': metadata.get('/Creator', ''),
                'producer': metadata.get('/Producer', ''),
                'creation_date': str(metadata.get('/CreationDate', '')),
                'num_pages': len(pages)
            }
        }
    
    def extract_abstract(self, text: str) -> Optional[str]:
        """
        Extract abstract section from paper text
        Simple heuristic-based extraction
        """
        text_lower = text.lower()
        
        # Common abstract markers
        abstract_markers = [
            'abstract',
            'summary',
            'introduction',
        ]
        
        lines = text.split('\n')
        abstract_start = -1
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(marker in line_lower for marker in abstract_markers):
                if 'abstract' in line_lower or 'summary' in line_lower:
                    abstract_start = i
                    break
        
        if abstract_start >= 0:
            # Extract next 10-20 lines as abstract
            abstract_lines = []
            for line in lines[abstract_start + 1:abstract_start + 20]:
                line = line.strip()
                if line and len(line) > 20:
                    abstract_lines.append(line)
                elif abstract_lines and len(line) < 10:
                    break
            
            return ' '.join(abstract_lines[:15])  # Limit length
        
        return None

