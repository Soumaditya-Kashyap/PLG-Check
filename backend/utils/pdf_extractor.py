import PyPDF2
import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            logger.info("Starting PDF text extraction...")
            reader = PyPDF2.PdfReader(pdf_file)
            
            total_pages = len(reader.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            text = ""
            
            for i, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logger.info(f"Processed page {i}/{total_pages} - {len(page_text)} characters")
                else:
                    logger.warning(f"No text found on page {i}")
            
            total_chars = len(text)
            logger.info(f"PDF extraction complete: {total_chars:,} total characters")
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove page numbers and headers/footers (basic approach)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be page numbers
            if len(line) > 10:
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines).strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better search"""
        logger.info("Chunking text into segments...")
        logger.info(f"Chunk size: {chunk_size} words, Overlap: {overlap} words")
        
        words = text.split()
        total_words = len(words)
        logger.info(f"Total words to process: {total_words:,}")
        
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk) > 50:  # Only include substantial chunks
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} text chunks for analysis")
        if chunks:
            avg_length = sum(len(chunk) for chunk in chunks) // len(chunks)
            logger.info(f"Average chunk length: {avg_length} characters")
        
        return chunks
