"""
Bibliographic Metadata Extraction Module

This module extracts structured bibliographic information from academic paper text
using pattern matching and heuristic rules. It identifies key metadata fields
such as authors, venue, publication year, DOI, keywords, and citation counts.

Key Features:
- Author name extraction from header sections
- Publication venue and journal identification
- Year extraction (1900-2099)
- DOI detection using standard patterns
- Keyword extraction from designated sections
- Email address extraction
- Reference citation counting

Extraction Methods:
- Pattern-based matching for structured fields
- Heuristic analysis of document structure
- Header section analysis (first 50 lines)
- Section-specific extraction (abstract, keywords, references)

Author: Research Team
Date: 2024
"""
import re
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts bibliographic metadata from paper text"""
    
    def __init__(self):
        # Common patterns for metadata extraction
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.doi_pattern = r'doi:\s*([\d\.]+/[\S]+)|10\.\d{4,}/[\S]+'
        self.year_pattern = r'\b(19|20)\d{2}\b'
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
    def extract(self, text: str, pdf_metadata: Dict = None) -> Dict:
        """
        Extract comprehensive metadata from paper text
        
        Args:
            text: Full text of the paper
            pdf_metadata: Metadata from PDF parser
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'title': '',
            'authors': [],
            'abstract': '',
            'keywords': [],
            'doi': None,
            'year': None,
            'venue': None,
            'affiliations': [],
            'emails': [],
            'references_count': 0,
            'citations': []
        }
        
        # Merge PDF metadata
        if pdf_metadata:
            metadata['title'] = pdf_metadata.get('title', '') or metadata['title']
            if pdf_metadata.get('author'):
                metadata['authors'] = self._parse_authors(pdf_metadata['author'])
        
        # Extract from text
        lines = text.split('\n')[:50]  # Analyze first 50 lines (usually header)
        
        # Extract title (usually first non-empty line)
        if not metadata['title']:
            metadata['title'] = self._extract_title(lines)
        
        # Extract authors
        if not metadata['authors']:
            metadata['authors'] = self._extract_authors(lines)
        
        # Extract abstract
        metadata['abstract'] = self._extract_abstract(text)
        
        # Extract keywords
        metadata['keywords'] = self._extract_keywords(text)
        
        # Extract DOI
        metadata['doi'] = self._extract_doi(text)
        
        # Extract year
        metadata['year'] = self._extract_year(text)
        
        # Extract venue/journal
        metadata['venue'] = self._extract_venue(lines)
        
        # Extract emails
        metadata['emails'] = self._extract_emails(text)
        
        # Count references
        metadata['references_count'] = self._count_references(text)
        
        # Extract citations (simplified)
        metadata['citations'] = self._extract_citations(text)
        
        return metadata
    
    def _extract_title(self, lines: List[str]) -> str:
        """Extract title from first few lines"""
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # Check if it looks like a title (no common words at start)
                if not line.lower().startswith(('abstract', 'introduction', 'keywords')):
                    return line
        return ''
    
    def _extract_authors(self, lines: List[str]) -> List[str]:
        """Extract author names"""
        authors = []
        
        # Look for lines with common author patterns
        for line in lines[:20]:
            line = line.strip()
            
            # Pattern: Name, Name, Name (common format)
            if ',' in line and len(line) > 10 and len(line) < 200:
                # Check if it contains author-like patterns
                if re.search(r'\b[A-Z][a-z]+\s+[A-Z]', line):
                    parts = line.split(',')
                    authors.extend([p.strip() for p in parts if len(p.strip()) > 3])
                    if len(authors) >= 1:
                        break
        
        return authors[:10]  # Limit to 10 authors
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section"""
        text_lower = text.lower()
        abstract_start = -1
        
        # Find abstract marker
        markers = ['abstract', 'summary']
        for marker in markers:
            idx = text_lower.find(marker)
            if idx >= 0:
                abstract_start = idx
                break
        
        if abstract_start >= 0:
            # Extract next ~500 characters
            abstract_text = text[abstract_start:abstract_start + 800]
            # Clean up
            abstract_text = re.sub(r'abstract\s*', '', abstract_text, flags=re.IGNORECASE)
            abstract_text = abstract_text.strip()[:500]  # Limit length
            return abstract_text
        
        # Fallback: first 300 characters
        return text[:300].strip()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords section"""
        keywords = []
        text_lower = text.lower()
        
        # Find keywords section
        keyword_markers = ['keywords:', 'key words:', 'index terms:']
        for marker in keyword_markers:
            idx = text_lower.find(marker)
            if idx >= 0:
                keyword_section = text[idx:idx + 200]
                # Extract comma or semicolon separated keywords
                keywords_text = re.sub(marker, '', keyword_section, flags=re.IGNORECASE)
                keywords = [k.strip() for k in re.split(r'[,;]', keywords_text) if len(k.strip()) > 2]
                keywords = keywords[:10]  # Limit
                break
        
        return keywords
    
    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI"""
        matches = re.findall(self.doi_pattern, text, re.IGNORECASE)
        if matches:
            doi = matches[0] if isinstance(matches[0], str) else matches[0][0] if matches[0][0] else matches[0][1]
            return doi.strip()
        return None
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract publication year"""
        # Look for 4-digit years (1900-2099)
        years = re.findall(self.year_pattern, text[:2000])  # Check first 2000 chars
        if years:
            # Return most recent year (usually publication year)
            years_int = [int(y) for y in years if 1900 <= int(y) <= 2099]
            if years_int:
                return max(years_int)
        return None
    
    def _extract_venue(self, lines: List[str]) -> Optional[str]:
        """Extract venue/journal name"""
        # Common venue patterns
        venue_keywords = ['proceedings', 'journal', 'conference', 'workshop', 'symposium']
        
        for line in lines[:30]:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in venue_keywords):
                return line.strip()[:100]
        
        return None
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        emails = re.findall(self.email_pattern, text[:5000])  # Check first 5000 chars
        return list(set(emails))[:5]  # Remove duplicates, limit to 5
    
    def _count_references(self, text: str) -> int:
        """Count number of references"""
        # Look for references section
        ref_markers = ['references', 'bibliography', 'works cited']
        text_lower = text.lower()
        
        for marker in ref_markers:
            idx = text_lower.rfind(marker)  # Start from end
            if idx >= 0:
                ref_section = text[idx:]
                # Count reference-like patterns [1], (Author, Year), etc.
                ref_count = len(re.findall(r'\[\d+\]|\(\w+,\s*\d{4}\)', ref_section))
                return max(ref_count, 0)
        
        return 0
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation mentions (simplified)"""
        # Extract patterns like [1], [2-5], (Author, 2020), etc.
        citations = []
        
        # Pattern 1: [1], [2], etc.
        citations.extend(re.findall(r'\[(\d+)\]', text[:5000]))
        
        # Pattern 2: (Author, Year)
        citations.extend(re.findall(r'\(([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*,\s+\d{4})\)', text[:5000]))
        
        return citations[:20]  # Limit
    
    def _parse_authors(self, author_string: str) -> List[str]:
        """Parse author string into list"""
        if not author_string:
            return []
        
        # Split by common delimiters
        authors = re.split(r'[,;]|and', author_string, flags=re.IGNORECASE)
        authors = [a.strip() for a in authors if len(a.strip()) > 2]
        return authors

