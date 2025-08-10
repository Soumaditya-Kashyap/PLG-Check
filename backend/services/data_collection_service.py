"""
Data Collection Service - Orchestrates LLM keyword extraction, ArXiv search, and web scraping
Focus on robust data collection for later plagiarism analysis
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from .llm_service import LLMService
from .arxiv_service import ArxivService
from .tavily_service import TavilyService
from utils.smart_text_processor import SmartTextProcessor

logger = logging.getLogger(__name__)

class DataCollectionService:
    def __init__(self):
        """Initialize all sub-services"""
        try:
            self.llm_service = LLMService()
            self.arxiv_service = ArxivService()
            self.tavily_service = TavilyService()
            self.text_processor = SmartTextProcessor()
            
            # Set up main storage directories (separate from uploads)
            from config import Config
            self.storage_base = os.path.join(os.getcwd(), 'scraped_data')  # Main scraped data folder
            self.results_base = os.path.join(os.getcwd(), 'results')      # Results folder for chunks/keywords
            
            # Create base directories
            os.makedirs(self.storage_base, exist_ok=True)
            os.makedirs(self.results_base, exist_ok=True)
            
            logger.info("Data Collection Service initialized successfully")
            logger.info(f"Scraped data will be stored in: {self.storage_base}")
            logger.info(f"Chunks/Keywords will be stored in: {self.results_base}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Data Collection Service: {e}")
            raise

    def _safe_json_parse(self, llm_output: str) -> Optional[List[Dict]]:
        """
        Cleans up LLM output and safely parses JSON (like leader's approach)
        """
        text = llm_output.strip()
        text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip())

        match = re.search(r"\[.*\]|\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def process_pdf_comprehensive(self, pdf_path: str, max_arxiv_papers: int = 5, max_web_results_per_chunk: int = 2) -> Dict[str, Any]:
        """
        Complete pipeline: PDF → Smart Chunks → LLM Keywords → ArXiv + Web Search → Storage
        Creates organized folder structure like your leader's approach
        """
        try:
            start_time = datetime.now()
            
            # Get file name without extension for folder naming
            pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            document_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pdf_filename}"
            
            logger.info(f"Starting comprehensive processing of: {pdf_filename}")
            logger.info(f"Document ID: {document_id}")
            
            # Create document-specific folders
            doc_scraped_folder = os.path.join(self.storage_base, document_id)
            doc_arxiv_folder = os.path.join(doc_scraped_folder, 'arxiv_papers')
            doc_web_folder = os.path.join(doc_scraped_folder, 'web_content')
            
            os.makedirs(doc_scraped_folder, exist_ok=True)
            os.makedirs(doc_arxiv_folder, exist_ok=True)
            os.makedirs(doc_web_folder, exist_ok=True)
            
            # Step 1: Extract and chunk PDF using smart processing
            logger.info("Step 1: Extracting and chunking PDF...")
            chunks = self.text_processor.extract_text_with_smart_chunking(pdf_path)
            
            if not chunks:
                return {
                    'success': False,
                    'error': 'Failed to extract chunks from PDF',
                    'pdf_path': pdf_path
                }
            
            logger.info(f"Extracted {len(chunks)} semantic chunks")
            
            # Step 2: Save chunks.json in results folder (like leader's approach)
            chunks_json_path = os.path.join(self.results_base, f"{pdf_filename}.chunks.json")
            with open(chunks_json_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved chunks to: {chunks_json_path}")
            
            # Step 3: Extract keywords from sections using LLM (like leader's approach)
            logger.info("Step 2: Extracting keywords using LLM...")
            section_keywords = {}
            keyword_results = []
            
            if self.llm_service.is_available():
                # Process sections in batches like leader's code
                filtered_chunks = [chunk for chunk in chunks if 'reference' not in chunk.get('section_type', '').lower()]
                
                for i in range(0, len(filtered_chunks), 5):  # Process 5 sections at a time
                    batch = filtered_chunks[i:i+5]
                    batch_text = ""
                    
                    for chunk in batch:
                        section_name = chunk.get('section_type', 'Unknown')
                        section_text = chunk.get('text', '')
                        batch_text += f"Section: {section_name}\nText:\n{section_text}\n\n"
                    
                    # Use similar prompt as leader's approach
                    prompt = (
                        "You are a research assistant. For each section below, extract:\n"
                        "1. **Top 3 most relevant keywords** (single words, directly tied to the section context)\n"
                        "2. **Top 3 most relevant key phrases** (multi-word, directly tied to the section context)\n"
                        "3. **All important key points** (bullet points, capturing every crucial idea or fact in the section)\n\n"
                        "Return ONLY valid JSON in the format:\n"
                        "[\n"
                        "  {\n"
                        '    "section": "section name",\n'
                        '    "keywords": ["word1", "word2", "word3"],\n'
                        '    "key_phrases": ["phrase1", "phrase2", "phrase3"],\n'
                        '    "key_points": ["point 1", "point 2", "point 3"]\n'
                        "  }\n"
                        "]\n"
                        f"Sections:\n{batch_text}"
                    )
                    
                    try:
                        response = self.llm_service.model.generate_content(prompt)
                        if response and response.text:
                            # Parse JSON response like leader's safe_json_parse
                            parsed = self._safe_json_parse(response.text)
                            if parsed and isinstance(parsed, list):
                                keyword_results.extend(parsed)
                            else:
                                # Fallback for failed parsing
                                for chunk in batch:
                                    section_name = chunk.get('section_type', 'Unknown')
                                    fallback_keywords = self.llm_service._fallback_keyword_extraction(chunk.get('text', ''))
                                    keyword_results.append({
                                        'section': section_name,
                                        'keywords': fallback_keywords[:3],
                                        'key_phrases': [],
                                        'key_points': []
                                    })
                    except Exception as e:
                        logger.warning(f"LLM processing failed for batch: {e}")
                        # Use fallback for this batch
                        for chunk in batch:
                            section_name = chunk.get('section_type', 'Unknown')
                            fallback_keywords = self.llm_service._fallback_keyword_extraction(chunk.get('text', ''))
                            keyword_results.append({
                                'section': section_name,
                                'keywords': fallback_keywords[:3],
                                'key_phrases': [],
                                'key_points': []
                            })
                
                logger.info(f"Processed keywords for {len(keyword_results)} sections")
            else:
                logger.warning("LLM service not available, using fallback keyword extraction")
                for chunk in chunks:
                    section_name = chunk.get('section_type', 'Unknown')
                    fallback_keywords = self.llm_service._fallback_keyword_extraction(chunk.get('text', ''))
                    keyword_results.append({
                        'section': section_name,
                        'keywords': fallback_keywords[:3],
                        'key_phrases': [],
                        'key_points': []
                    })
            
            # Step 4: Save keywords.json in results folder (like leader's approach)
            keywords_json_path = os.path.join(self.results_base, f"{pdf_filename}.keywords.json")
            with open(keywords_json_path, 'w', encoding='utf-8') as f:
                json.dump(keyword_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved keywords to: {keywords_json_path}")
            
            # Step 5: Search ArXiv using combined keywords
            logger.info("Step 3: Searching ArXiv papers...")
            arxiv_results = {'success': False, 'papers_downloaded': 0, 'results': []}
            
            if keyword_results:
                # Combine all keywords for ArXiv search
                all_keywords = []
                for result in keyword_results:
                    all_keywords.extend(result.get('keywords', []))
                
                # Remove duplicates and take top keywords
                unique_keywords = list(set(all_keywords))[:10]
                
                if unique_keywords:
                    arxiv_results = self.arxiv_service.search_and_download(unique_keywords, max_arxiv_papers, doc_arxiv_folder)
                    logger.info(f"ArXiv search completed: {arxiv_results.get('papers_downloaded', 0)} papers downloaded")
            
            # Step 6: Search web content using chunk text
            logger.info("Step 4: Searching and extracting web content...")
            web_results = []
            
            # Use chunk text for web search (more contextual than just keywords)
            chunk_texts = [chunk.get('text', '') for chunk in chunks if chunk.get('text', '').strip()]
            
            if chunk_texts:
                web_results = self.tavily_service.process_chunks_batch(chunk_texts[:5], max_web_results_per_chunk, doc_web_folder)
                total_web_content = sum(r.get('content_extracted', 0) for r in web_results)
                logger.info(f"Web search completed: {total_web_content} content pieces extracted")
            
            # Step 7: Compile and save comprehensive results
            logger.info("Step 5: Compiling and saving results...")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            comprehensive_results = {
                'processing_info': {
                    'pdf_path': pdf_path,
                    'pdf_name': os.path.basename(pdf_path),
                    'document_id': document_id,
                    'processing_date': start_time.isoformat(),
                    'processing_time_seconds': processing_time,
                    'success': True
                },
                'file_structure': {
                    'chunks_json': chunks_json_path,
                    'keywords_json': keywords_json_path,
                    'scraped_data_folder': doc_scraped_folder,
                    'arxiv_papers_folder': doc_arxiv_folder,
                    'web_content_folder': doc_web_folder
                },
                'chunks_analysis': {
                    'total_chunks': len(chunks),
                    'chunks_data': chunks
                },
                'keywords_extraction': {
                    'llm_available': self.llm_service.is_available(),
                    'sections_processed': len(keyword_results),
                    'keyword_results': keyword_results,
                    'combined_keywords': list(set([kw for result in keyword_results for kw in result.get('keywords', [])]))
                },
                'arxiv_collection': arxiv_results,
                'web_collection': {
                    'chunks_processed': len(web_results),
                    'total_results': sum(r.get('results_found', 0) for r in web_results),
                    'total_content_extracted': sum(r.get('content_extracted', 0) for r in web_results),
                    'detailed_results': web_results
                },
                'storage_info': {
                    'document_folder': doc_scraped_folder,
                    'arxiv_folder': doc_arxiv_folder,
                    'web_folder': doc_web_folder,
                    'results_folder': self.results_base
                }
            }
            
            # Save comprehensive results to document folder
            result_filename = f"collection_summary.json"
            result_path = os.path.join(doc_scraped_folder, result_filename)
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
            
            comprehensive_results['result_file'] = result_path
            
            logger.info(f"Comprehensive processing completed in {processing_time:.1f}s")
            logger.info(f"Results saved to: {result_path}")
            logger.info(f"Folder structure created:")
            logger.info(f"  📁 {doc_scraped_folder}/")
            logger.info(f"    ├── arxiv_papers/ ({arxiv_results.get('papers_downloaded', 0)} papers)")
            logger.info(f"    ├── web_content/ ({sum(r.get('content_extracted', 0) for r in web_results)} files)")
            logger.info(f"    └── collection_summary.json")
            logger.info(f"  📁 {self.results_base}/")
            logger.info(f"    ├── {pdf_filename}.chunks.json")
            logger.info(f"    └── {pdf_filename}.keywords.json")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive PDF processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'pdf_path': pdf_path,
                'processing_date': datetime.now().isoformat()
            }

    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of all data collection activities"""
        try:
            # Get collection results files
            result_files = []
            chunks_files = []
            keywords_files = []
            scraped_folders = []
            
            if os.path.exists(self.results_base):
                all_files = os.listdir(self.results_base)
                chunks_files = [f for f in all_files if f.endswith('.chunks.json')]
                keywords_files = [f for f in all_files if f.endswith('.keywords.json')]
            
            if os.path.exists(self.storage_base):
                scraped_folders = [d for d in os.listdir(self.storage_base) if os.path.isdir(os.path.join(self.storage_base, d))]
            
            # Get ArXiv and web stats from each document folder
            total_arxiv_papers = 0
            total_web_content = 0
            
            for folder in scraped_folders:
                arxiv_folder = os.path.join(self.storage_base, folder, 'arxiv_papers')
                web_folder = os.path.join(self.storage_base, folder, 'web_content')
                
                if os.path.exists(arxiv_folder):
                    arxiv_files = [f for f in os.listdir(arxiv_folder) if f.endswith('.pdf')]
                    total_arxiv_papers += len(arxiv_files)
                
                if os.path.exists(web_folder):
                    web_files = [f for f in os.listdir(web_folder) if f.endswith('.txt')]
                    total_web_content += len(web_files)
            
            # Get LLM model info
            llm_info = self.llm_service.get_model_info()
            
            return {
                'collection_overview': {
                    'total_documents_processed': len(scraped_folders),
                    'total_chunks_files': len(chunks_files),
                    'total_keywords_files': len(keywords_files),
                    'last_activity': datetime.now().isoformat()
                },
                'content_summary': {
                    'total_arxiv_papers': total_arxiv_papers,
                    'total_web_content_files': total_web_content,
                    'processed_documents': scraped_folders
                },
                'llm_service': llm_info,
                'storage_directories': {
                    'scraped_data_base': self.storage_base,
                    'results_base': self.results_base,
                    'document_folders': scraped_folders
                },
                'file_organization': {
                    'chunks_files': chunks_files,
                    'keywords_files': keywords_files,
                    'structure': {
                        'scraped_data/': 'Contains document-specific folders with arxiv_papers/ and web_content/',
                        'results/': 'Contains .chunks.json and .keywords.json files for each processed PDF'
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting collection summary: {e}")
            return {'error': str(e)}

    def process_sections_individually(self, pdf_path: str, max_arxiv_per_section: int = 2, max_web_per_section: int = 2) -> Dict[str, Any]:
        """
        Alternative approach: Process each section individually for more targeted results
        
        Args:
            pdf_path: Path to the PDF file
            max_arxiv_per_section: Max ArXiv papers per section
            max_web_per_section: Max web results per section
            
        Returns:
            Section-by-section results
        """
        try:
            logger.info(f"Starting section-by-section processing of: {os.path.basename(pdf_path)}")
            
            # Extract chunks
            chunks = self.text_processor.extract_text_with_smart_chunking(pdf_path)
            
            if not chunks:
                return {'success': False, 'error': 'Failed to extract chunks'}
            
            section_results = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing section {i+1}/{len(chunks)}")
                
                section_text = chunk.get('text', '')
                section_type = chunk.get('section_type', 'unknown')
                
                if not section_text.strip():
                    continue
                
                # Extract keywords for this section
                keywords = []
                if self.llm_service.is_available():
                    keywords = self.llm_service.extract_keywords_from_section(section_text, section_type)
                else:
                    keywords = self.llm_service._fallback_keyword_extraction(section_text)
                
                # Search ArXiv with section keywords
                arxiv_result = {'success': False, 'papers_downloaded': 0}
                if keywords:
                    arxiv_result = self.arxiv_service.search_and_download(keywords, max_arxiv_per_section)
                
                # Search web with section text
                web_result = self.tavily_service.search_and_store(section_text, max_web_per_section)
                
                section_results.append({
                    'section_index': i,
                    'section_type': section_type,
                    'text_length': len(section_text),
                    'keywords': keywords,
                    'arxiv_results': arxiv_result,
                    'web_results': web_result
                })
            
            return {
                'success': True,
                'pdf_path': pdf_path,
                'processing_mode': 'section_by_section',
                'total_sections': len(section_results),
                'section_results': section_results,
                'processing_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in section-by-section processing: {e}")
            return {'success': False, 'error': str(e)}
