import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SmartTextProcessor:
    """Enhanced text processor using semantic chunking based on document structure"""
    
    def __init__(self):
        self.known_headings = {
            "abstract", "introduction", "related work", "background", "preliminaries",
            "method", "methodology", "approach", "experiment", "evaluation", "experiments",
            "results", "analysis", "discussion", "conclusion", "conclusions", "future work", 
            "references", "bibliography", "acknowledgments", "acknowledgements"
        }
        
        self.numbering_regexes = [
            r'^\s*\d+(\.\d+)*\s+[A-Z]',  # 1. Introduction, 1.1 Background
            r'^\s*[IVXLC]+\.\s+[A-Z]',   # I. Introduction, II. Background
            r'^\s*[A-Z]\.\s+[A-Z]',      # A. Introduction, B. Background
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract full text for fallback purposes"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
            return full_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def extract_blocks_with_layout(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with layout information"""
        try:
            doc = fitz.open(pdf_path)
            all_blocks = []
            
            for page_num, page in enumerate(doc, start=1):
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        text = "".join([span["text"] for span in line["spans"]]).strip()
                        if not text:
                            continue
                        font_sizes = [span["size"] for span in line["spans"]]
                        avg_font = mean(font_sizes) if font_sizes else 12
                        all_blocks.append({
                            "text": text,
                            "font_size": avg_font,
                            "y0": block["bbox"][1],
                            "page": page_num
                        })
            doc.close()
            return all_blocks
        except Exception as e:
            logger.error(f"Error extracting blocks with layout: {str(e)}")
            return []

    def is_all_caps(self, text: str) -> bool:
        """Check if text is in all caps"""
        letters = [c for c in text if c.isalpha()]
        return len(letters) > 3 and all(c.isupper() for c in letters)

    def is_title_case(self, text: str) -> bool:
        """Check if text is in title case"""
        return text.istitle()

    def matches_numbering(self, text: str) -> bool:
        """Check if text matches section numbering patterns"""
        return any(re.match(rgx, text) for rgx in self.numbering_regexes)

    def is_known_heading(self, text: str) -> bool:
        """Check if text matches known academic section headings"""
        text_lower = text.lower().strip()
        return any(heading in text_lower for heading in self.known_headings)

    def compute_scores(self, blocks: List[Dict]) -> List[Dict]:
        """Score blocks for heading likelihood"""
        if not blocks:
            return blocks
            
        font_sizes = [b["font_size"] for b in blocks]
        avg_font = mean(font_sizes) if font_sizes else 12
        max_font = max(font_sizes) if font_sizes else 12

        for i, block in enumerate(blocks):
            score = 0
            text = block["text"]

            # Font size scoring
            if block["font_size"] >= avg_font * 1.3:
                score += 4
            elif block["font_size"] >= avg_font * 1.15:
                score += 3
            elif block["font_size"] >= avg_font * 1.05:
                score += 1

            # Capitalization patterns
            if self.is_all_caps(text):
                score += 2
            if self.is_title_case(text):
                score += 1

            # Numbered sections
            if self.matches_numbering(text):
                score += 3

            # Known academic headings
            if self.is_known_heading(text):
                score += 3

            # Length considerations
            if len(text.split()) <= 10:  # Short text more likely to be heading
                score += 1

            # Vertical spacing
            if i > 0:
                y_gap = abs(block["y0"] - blocks[i - 1]["y0"])
                if y_gap > 25:
                    score += 2
                elif y_gap > 15:
                    score += 1

            block["score"] = score
            
        return blocks

    def create_semantic_chunks(self, pdf_path: str, threshold: int = 3) -> List[Dict[str, Any]]:
        """Create high-quality semantic chunks like leader's format - clean, longer, meaningful content"""
        logger.info(f"🧠 Creating high-quality semantic chunks for: {pdf_path}")
        
        try:
            # Extract text with better structure preservation
            doc = fitz.open(pdf_path)
            all_sections = []
            current_section = {
                "title": "Document Content",
                "content": "",
                "page": 1
            }
            
            for page_num, page in enumerate(doc, start=1):
                # Get text with layout information
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" not in block:
                        continue
                    
                    # Extract text from block
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        
                        if line_text.strip():
                            block_text += line_text.strip() + " "
                    
                    if not block_text.strip():
                        continue
                    
                    # Check if this block is a section heading
                    if self._is_quality_section_heading(block_text.strip()):
                        # Save previous section if it has substantial content
                        if current_section["content"].strip() and len(current_section["content"].strip()) > 150:
                            all_sections.append(current_section.copy())
                        
                        # Start new section
                        current_section = {
                            "title": self._clean_section_title(block_text.strip()),
                            "content": "",
                            "page": page_num
                        }
                        logger.debug(f"📝 New section: {current_section['title']}")
                    else:
                        # Add content to current section
                        clean_content = self._clean_content_text(block_text)
                        if clean_content:
                            current_section["content"] += clean_content + " "
            
            # Add final section
            if current_section["content"].strip() and len(current_section["content"].strip()) > 150:
                all_sections.append(current_section)
            
            doc.close()
            
            # Create high-quality chunks from sections
            final_chunks = self._create_leader_quality_chunks(all_sections)
            
            logger.info(f"✅ Created {len(final_chunks)} leader-quality semantic chunks")
            return final_chunks if final_chunks else self._fallback_quality_chunking(pdf_path)

        except Exception as e:
            logger.error(f"Quality chunking failed: {e}, using fallback")
            return self._fallback_quality_chunking(pdf_path)

    def _is_quality_section_heading(self, text: str) -> bool:
        """Improved section heading detection for better quality chunks"""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Skip very long lines (likely content, not headings)
        if len(text_clean) > 120:
            return False
        
        # Skip very short lines
        if len(text_clean) < 4:
            return False
        
        # Priority 1: Numbered sections (1. Introduction, 2.1 Background, etc.)
        numbering_patterns = [
            r'^\d+\.\s+[A-Z]',  # 1. Introduction
            r'^\d+\.\d+\s+[A-Z]',  # 2.1 Background
            r'^\d+\.\d+\.\d+\s+[A-Z]',  # 2.1.1 Details
            r'^[IVX]+\.\s+[A-Z]',  # I. Introduction (Roman numerals)
            r'^[A-Z]\.\s+[A-Z]',  # A. Section
        ]
        
        for pattern in numbering_patterns:
            if re.match(pattern, text_clean):
                return True
        
        # Priority 2: Known academic section headers
        academic_sections = [
            'abstract', 'introduction', 'background', 'literature review',
            'methodology', 'methods', 'approach', 'implementation',
            'results', 'findings', 'evaluation', 'experiments',
            'discussion', 'analysis', 'conclusion', 'conclusions',
            'future work', 'references', 'bibliography', 'acknowledgments',
            'related work', 'case study', 'data collection', 'survey'
        ]
        
        for section in academic_sections:
            if text_lower == section or text_lower.startswith(section + ' '):
                return True
        
        # Priority 3: Formatting clues
        # All caps (common for headings)
        if text_clean.isupper() and 4 <= len(text_clean) <= 60:
            return True
        
        # Title case without ending punctuation
        if (text_clean.istitle() and 
            not text_clean.endswith('.') and 
            not text_clean.endswith(',') and
            5 <= len(text_clean) <= 80):
            return True
        
        return False
    
    def _clean_section_title(self, title: str) -> str:
        """Clean up section titles"""
        # Remove numbering
        title = re.sub(r'^\s*\d+(\.\d+)*\s*', '', title)
        title = re.sub(r'^\s*[IVXLC]+\.\s*', '', title)
        title = re.sub(r'^\s*[A-Z]\.\s*', '', title)
        
        # Clean up
        title = title.strip()
        if not title:
            return "Content"
        
        # Convert to title case if all caps
        if title.isupper():
            title = title.title()
        
        # Limit length
        if len(title) > 60:
            title = title[:60] + "..."
        
        return title
    
    def _clean_content_text(self, text: str) -> str:
        """Clean content text for better readability"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix hyphenated line breaks
        text = re.sub(r'-\s+', '', text)
        
        # Fix sentence spacing
        text = re.sub(r'\.(\w)', r'. \1', text)
        
        # Remove isolated numbers/page numbers
        text = re.sub(r'\b\d+\b(?=\s*$)', '', text)
        
        # Remove repeated characters
        text = re.sub(r'(.)\1{4,}', r'\1', text)
        
        return text.strip()
    
    def _create_leader_quality_chunks(self, sections: List[Dict]) -> List[Dict[str, Any]]:
        """Create high-quality chunks similar to leader's format"""
        chunks = []
        
        for section in sections:
            title = section["title"]
            content = section["content"].strip()
            page = section["page"]
            
            if len(content) < 200:  # Skip very short sections
                continue
            
            # If section is reasonably sized, keep as one chunk
            if len(content) <= 2500:
                chunks.append({
                    "text": self._format_chunk_text(content),
                    "section": title,
                    "page": page
                })
            else:
                # Split long sections intelligently
                sub_chunks = self._split_long_section(content, title, page)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _format_chunk_text(self, text: str) -> str:
        """Format chunk text for maximum readability and quality"""
        # Split into sentences and rejoin properly
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean each sentence
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short fragments
                # Ensure proper capitalization
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                clean_sentences.append(sentence)
        
        # Rejoin with proper spacing
        formatted_text = ' '.join(clean_sentences)
        
        # Final cleanup
        formatted_text = re.sub(r'\s+', ' ', formatted_text)
        
        return formatted_text.strip()
    
    def _split_long_section(self, content: str, section_title: str, page: int) -> List[Dict[str, Any]]:
        """Split long sections at natural boundaries"""
        # Split into paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_count = 1
        
        for para in paragraphs:
            # If adding this paragraph would make chunk too long, save current chunk
            if len(current_chunk + para) > 2000 and current_chunk.strip():
                chunks.append({
                    "text": self._format_chunk_text(current_chunk),
                    "section": f"{section_title} (Part {chunk_count})",
                    "page": page
                })
                current_chunk = para + "\n\n"
                chunk_count += 1
            else:
                current_chunk += para + "\n\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunk_title = f"{section_title} (Part {chunk_count})" if chunk_count > 1 else section_title
            chunks.append({
                "text": self._format_chunk_text(current_chunk),
                "section": chunk_title,
                "page": page
            })
        
        return chunks
    
    def _fallback_quality_chunking(self, pdf_path: str) -> List[Dict[str, Any]]:
        """High-quality fallback chunking"""
        try:
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                return []
            
            # Clean the entire text
            text = self._clean_content_text(text)
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
            
            chunks = []
            current_chunk = ""
            chunk_num = 1
            
            for para in paragraphs:
                if len(current_chunk + para) > 2000 and current_chunk.strip():
                    chunks.append({
                        "text": self._format_chunk_text(current_chunk),
                        "section": f"Document Content (Part {chunk_num})",
                        "page": 1
                    })
                    current_chunk = para + "\n\n"
                    chunk_num += 1
                else:
                    current_chunk += para + "\n\n"
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    "text": self._format_chunk_text(current_chunk),
                    "section": f"Document Content (Part {chunk_num})" if chunk_num > 1 else "Document Content",
                    "page": 1
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Fallback chunking failed: {str(e)}")
            return []

    def clean_text_block(self, text: str) -> str:
        """Clean and normalize text block"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common PDF extraction issues
        text = re.sub(r'-\s*\n\s*', '', text)  # Remove hyphenated line breaks
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        # Remove extra spaces
        text = text.strip()
        return text

    def create_clean_chunks_from_section(self, content: str, section_name: str, page: int) -> List[Dict[str, Any]]:
        """Create clean, sentence-aware chunks from a section"""
        chunks = []
        
        # Split content into paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        target_length = 1000  # Target length for chunks
        max_length = 1800     # Maximum length before forcing split
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed max length
            if current_chunk and len(current_chunk + paragraph) > max_length:
                # Save current chunk if it's substantial
                if len(current_chunk.strip()) > 50:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": section_name,
                        "chunk_index": len(chunks),
                        "type": "section_content",
                        "page": page,
                        "start_pos": 0,
                        "end_pos": len(current_chunk.strip())
                    })
                current_chunk = paragraph + "\n"
            else:
                current_chunk += paragraph + "\n"
                
                # If we've reached a good size, check for natural break
                if len(current_chunk) >= target_length:
                    # Look for a good breaking point (end of sentence + some content)
                    sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                    if len(sentences) > 2:  # At least 2 complete sentences
                        # Find a good break point (around 2/3 through)
                        break_point = len(sentences) * 2 // 3
                        chunk_text = ' '.join(sentences[:break_point]).strip()
                        remaining_text = ' '.join(sentences[break_point:]).strip()
                        
                        if len(chunk_text) > 100:  # Ensure chunk is substantial
                            chunks.append({
                                "text": chunk_text,
                                "section": section_name,
                                "chunk_index": len(chunks),
                                "type": "section_content",
                                "page": page,
                                "start_pos": 0,
                                "end_pos": len(chunk_text)
                            })
                            current_chunk = remaining_text + "\n" if remaining_text else ""
        
        # Add any remaining content as final chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 50:
            chunks.append({
                "text": current_chunk.strip(),
                "section": section_name,
                "chunk_index": len(chunks),
                "type": "section_content",
                "page": page,
                "start_pos": 0,
                "end_pos": len(current_chunk.strip())
            })
            
        return chunks

    def create_sentence_aware_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create sentence-aware chunks as fallback method"""
        logger.info("Using sentence-aware chunking")
        chunks = []
        
        # Clean the text
        text = self.clean_text_block(text)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_size = 800  # Target chunk size
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= chunk_size * 1.5:  # Allow some flexibility
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": "Unknown",
                        "chunk_index": len(chunks),
                        "type": "text_chunk",
                        "page": 1,
                        "start_pos": 0,
                        "end_pos": len(current_chunk.strip())
                    })
                current_chunk = sentence + " "
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "section": "Unknown",
                "chunk_index": len(chunks),
                "type": "text_chunk",
                "page": 1,
                "start_pos": 0,
                "end_pos": len(current_chunk.strip())
            })
            
        return chunks

    def create_simple_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Legacy fallback method - redirects to sentence-aware chunking"""
        logger.info("Redirecting to sentence-aware chunking")
        return self.create_sentence_aware_chunks(text)
        chunk_size = 300
        
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks.append({
                "text": chunk_text,
                "section": "Content",
                "chunk_index": len(chunks),
                "type": "simple",
                "page": 1,
                "start_pos": i * 5,  # Approximate
                "end_pos": (i + len(chunk_text.split())) * 5
            })
        
        return chunks

    def extract_title_from_text(self, text: str) -> str:
        """Extract document title from text"""
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if 10 < len(line) < 200 and not line.lower().startswith(('abstract', 'introduction')):
                # Check if it looks like a title (not too many special chars)
                special_char_ratio = sum(1 for c in line if not c.isalnum() and c not in ' -:') / len(line)
                if special_char_ratio < 0.3:
                    return line
        return "Document"

    def generate_search_queries(self, title: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate search queries from smart chunks"""
        queries = [title]
        
        # Use section-based queries
        section_chunks = {}
        for chunk in chunks:
            section = chunk.get("section", "Content")
            if section not in section_chunks:
                section_chunks[section] = []
            section_chunks[section].append(chunk)
        
        # Generate queries from each section
        for section, section_chunk_list in section_chunks.items():
            if section != "Content" and section_chunk_list:
                # Take the first chunk from each section
                chunk = section_chunk_list[0]
                text = chunk["text"]
                
                if len(text) > 100:
                    # Extract meaningful phrases
                    words = text.split()[:20]  # First 20 words
                    query = " ".join(words)
                    
                    # Clean up query
                    query = re.sub(r'[^\w\s]', ' ', query)
                    query = ' '.join(query.split())
                    
                    if len(query) > 20:
                        queries.append(query)
        
        # Add some general content queries
        content_chunks = [c for c in chunks if c.get("type") == "content"][:3]
        for chunk in content_chunks:
            text = chunk["text"]
            if len(text) > 200:
                words = text.split()[:15]
                query = " ".join(words)
                query = re.sub(r'[^\w\s]', ' ', query)
                query = ' '.join(query.split())
                if len(query) > 15:
                    queries.append(query)
        
        return list(set(queries))[:10]  # Remove duplicates, limit to 10

    def chunk_text_with_positions(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Main method to create chunks with positions - for compatibility"""
        return self.create_semantic_chunks(pdf_path)
