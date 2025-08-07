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
        """Create semantic chunks based on document structure"""
        logger.info(f"Creating semantic chunks for: {pdf_path}")
        
        try:
            blocks = self.extract_blocks_with_layout(pdf_path)
            if not blocks:
                logger.warning("No blocks extracted, falling back to simple chunking")
                full_text = self.extract_text_from_pdf(pdf_path)
                return self.create_simple_chunks(full_text)
                
            logger.info(f"Extracted {len(blocks)} text blocks")
            scored_blocks = self.compute_scores(blocks)
            sections = []
            current_section = None

            for block in scored_blocks:
                text = block["text"]
                score = block["score"]

                if score >= threshold:
                    # New heading found
                    if current_section and current_section["content"].strip():
                        sections.append(current_section)

                    # Clean up section title
                    title = text.strip()
                    title_lower = title.lower()
                    
                    # Determine section title
                    if any(known in title_lower for known in self.known_headings):
                        section_title = title.title()
                    else:
                        section_title = title

                    current_section = {
                        "section": section_title,
                        "content": "",
                        "type": "section",
                        "page": block["page"]
                    }
                    logger.debug(f"New section detected: {section_title}")
                else:
                    # Regular content
                    if current_section:
                        current_section["content"] += text + "\n"
                    else:
                        # Content without header
                        current_section = {
                            "section": "Document Content",
                            "content": text + "\n",
                            "type": "content",
                            "page": block["page"]
                        }

            # Add final section
            if current_section and current_section["content"].strip():
                sections.append(current_section)

            logger.info(f"Identified {len(sections)} sections")

            # Convert to chunks suitable for plagiarism detection
            chunks = []
            for i, section in enumerate(sections):
                content = section["content"].strip()
                if len(content) > 50:  # Only meaningful content
                    # Split large sections into smaller chunks
                    if len(content) > 2000:
                        sub_chunks = self.split_large_section(content, section["section"], section["page"])
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append({
                            "text": content,
                            "section": section["section"],
                            "chunk_index": len(chunks),
                            "type": section["type"],
                            "page": section["page"],
                            "start_pos": 0,  # For compatibility
                            "end_pos": len(content)
                        })

            logger.info(f"Created {len(chunks)} semantic chunks")
            return chunks if chunks else self.create_simple_chunks(self.extract_text_from_pdf(pdf_path))

        except Exception as e:
            logger.error(f"Smart chunking failed: {e}, falling back to simple chunking")
            full_text = self.extract_text_from_pdf(pdf_path)
            return self.create_simple_chunks(full_text)

    def split_large_section(self, content: str, section_name: str, page: int) -> List[Dict[str, Any]]:
        """Split large sections into smaller chunks while preserving context"""
        chunks = []
        
        # Try to split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', content)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < 1500:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": f"{section_name} (Part {len(chunks) + 1})",
                        "chunk_index": len(chunks),
                        "type": "section_part",
                        "page": page,
                        "start_pos": 0,
                        "end_pos": len(current_chunk.strip())
                    })
                current_chunk = sentence + " "
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "section": f"{section_name} (Part {len(chunks) + 1})" if chunks else section_name,
                "chunk_index": len(chunks),
                "type": "section_part",
                "page": page,
                "start_pos": 0,
                "end_pos": len(current_chunk.strip())
            })
            
        return chunks

    def create_simple_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Fallback simple chunking method"""
        logger.info("Using fallback simple chunking")
        chunks = []
        words = text.split()
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
