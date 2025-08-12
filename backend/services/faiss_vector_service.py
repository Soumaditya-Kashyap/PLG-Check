import os
import json
import logging
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import PyPDF2
from config import Config

logger = logging.getLogger(__name__)

class FAISSVectorService:
    def __init__(self, vector_db_path: str = "faiss_vector_db"):
        """
        Initialize FAISS Vector Database Service for plagiarism detection
        
        Args:
            vector_db_path: Path where FAISS database will be stored
        """
        self.vector_db_path = vector_db_path
        self.embedding_model_name = 'all-MiniLM-L6-v2'  # Good balance of speed and quality
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize embedding model
        logger.info(f"🧠 Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Separate FAISS indexes for different content types
        self.indexes = {
            'user_pdf': None,      # User uploaded PDF content
            'arxiv_papers': None,  # ArXiv research papers
            'web_content': None    # Web scraped content
        }
        
        # Separate metadata for each content type
        self.metadata = {
            'user_pdf': [],
            'arxiv_papers': [],
            'web_content': []
        }
        
        # Document tracking
        self.document_metadata = []  # Store metadata for each document
        
        # Create vector DB directory
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Try to load existing indexes
        self._load_existing_indexes()
        
    def _load_existing_indexes(self):
        """Load existing FAISS indexes and metadata if available"""
        try:
            # Load each index type separately
            for content_type in ['user_pdf', 'arxiv_papers', 'web_content']:
                index_path = os.path.join(self.vector_db_path, f"{content_type}_index.bin")
                metadata_path = os.path.join(self.vector_db_path, f"{content_type}_metadata.pkl")
                
                if os.path.exists(index_path) and os.path.exists(metadata_path):
                    logger.info(f"📂 Loading existing {content_type} FAISS index...")
                    self.indexes[content_type] = faiss.read_index(index_path)
                    
                    with open(metadata_path, 'rb') as f:
                        self.metadata[content_type] = pickle.load(f)
                    
                    logger.info(f"✅ Loaded {content_type} index with {self.indexes[content_type].ntotal} vectors")
                else:
                    logger.info(f"🔨 Creating new {content_type} FAISS index...")
                    self.indexes[content_type] = faiss.IndexFlatIP(self.embedding_dimension)
                    self.metadata[content_type] = []
            
            # Load document metadata
            doc_metadata_path = os.path.join(self.vector_db_path, "document_metadata.pkl")
            if os.path.exists(doc_metadata_path):
                with open(doc_metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            else:
                self.document_metadata = []
                
        except Exception as e:
            logger.error(f"❌ Error loading FAISS indexes: {e}")
            self._create_new_indexes()
    
    def _create_new_indexes(self):
        """Create new FAISS indexes for each content type"""
        for content_type in ['user_pdf', 'arxiv_papers', 'web_content']:
            self.indexes[content_type] = faiss.IndexFlatIP(self.embedding_dimension)
            self.metadata[content_type] = []
            logger.info(f"✅ Created new {content_type} index with dimension {self.embedding_dimension}")
        
        self.document_metadata = []
    
    def _save_indexes(self):
        """Save all FAISS indexes and metadata to disk"""
        try:
            # Save each index type separately
            for content_type in ['user_pdf', 'arxiv_papers', 'web_content']:
                index_path = os.path.join(self.vector_db_path, f"{content_type}_index.bin")
                metadata_path = os.path.join(self.vector_db_path, f"{content_type}_metadata.pkl")
                
                # Save FAISS index
                faiss.write_index(self.indexes[content_type], index_path)
                
                # Save metadata
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self.metadata[content_type], f)
                
                logger.info(f"💾 Saved {content_type} index with {self.indexes[content_type].ntotal} vectors")
            
            # Save document metadata
            doc_metadata_path = os.path.join(self.vector_db_path, "document_metadata.pkl")
            doc_metadata_to_save = {
                'document_metadata': self.document_metadata,
                'created_at': datetime.now().isoformat(),
                'model_name': self.embedding_model_name,
                'dimension': self.embedding_dimension
            }
            
            with open(doc_metadata_path, 'wb') as f:
                pickle.dump(doc_metadata_to_save, f)
            
            # Calculate total vectors across all indexes
            total_vectors = sum(index.ntotal for index in self.indexes.values())
            logger.info(f"💾 Saved all FAISS indexes with {total_vectors} total vectors")
            
        except Exception as e:
            logger.error(f"❌ Error saving FAISS indexes: {e}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            text_content = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better embedding coverage
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size // 2:  # If we found a good boundary
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            start = end - overlap if end < len(text) else end
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter very short chunks
    
    def process_scraped_data_folder(self, document_folder: str) -> Dict[str, Any]:
        """
        Process a complete document folder and add all content to appropriate FAISS indexes
        
        Args:
            document_folder: Path to document folder (e.g., scraped_data/3534056.3534943)
        """
        logger.info(f"🔍 Processing document folder: {document_folder}")
        
        results = {
            'document_id': os.path.basename(document_folder),
            'arxiv_papers_processed': 0,
            'web_content_processed': 0,
            'total_chunks_added': 0,
            'arxiv_chunks': 0,
            'web_chunks': 0,
            'processing_errors': []
        }
        
        # Process ArXiv papers (add to arxiv_papers index)
        arxiv_folder = os.path.join(document_folder, 'arxiv')
        if os.path.exists(arxiv_folder):
            arxiv_results = self._process_arxiv_folder(arxiv_folder, results['document_id'])
            results['arxiv_papers_processed'] = arxiv_results['arxiv_papers_processed']
            results['arxiv_chunks'] = arxiv_results['arxiv_chunks']
            results['processing_errors'].extend(arxiv_results['processing_errors'])
        
        # Process Web content (add to web_content index)
        web_folder = os.path.join(document_folder, 'web')
        if os.path.exists(web_folder):
            web_results = self._process_web_folder(web_folder, results['document_id'])
            results['web_content_processed'] = web_results['web_content_processed']
            results['web_chunks'] += web_results['web_chunks']
            results['processing_errors'].extend(web_results['processing_errors'])
        
        results['total_chunks_added'] = results['arxiv_chunks'] + results['web_chunks']
        
        # Save updated indexes
        if results['total_chunks_added'] > 0:
            self._save_indexes()
            logger.info(f"✅ Added {results['total_chunks_added']} chunks from document {results['document_id']} "
                       f"(ArXiv: {results['arxiv_chunks']}, Web: {results['web_chunks']})")
        
        return results
    
    def _process_arxiv_folder(self, arxiv_folder: str, document_id: str) -> Dict[str, Any]:
        """Process ArXiv papers from a document folder"""
        results = {
            'arxiv_papers_processed': 0,
            'arxiv_chunks': 0,
            'processing_errors': []
        }
        
        # Check for PDFs in pdfs folder
        pdfs_folder = os.path.join(arxiv_folder, 'pdfs')
        if not os.path.exists(pdfs_folder):
            logger.warning(f"⚠️ No PDFs folder found in {arxiv_folder}")
            return results
        
        # Process each PDF
        for pdf_file in os.listdir(pdfs_folder):
            if pdf_file.endswith('.pdf'):
                try:
                    pdf_path = os.path.join(pdfs_folder, pdf_file)
                    
                    # Extract text from PDF
                    pdf_text = self._extract_text_from_pdf(pdf_path)
                    if not pdf_text:
                        results['processing_errors'].append(f"No text extracted from {pdf_file}")
                        continue
                    
                    # Load metadata if available
                    metadata_folder = os.path.join(pdfs_folder, 'metadata')
                    metadata_file = pdf_file.replace('.pdf', '.json')
                    metadata_path = os.path.join(metadata_folder, metadata_file)
                    
                    paper_metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            paper_metadata = json.load(f)
                    
                    # Create chunks
                    chunks = self._chunk_text(pdf_text)
                    
                    # Add chunks to ArXiv FAISS index
                    for i, chunk in enumerate(chunks):
                        self._add_chunk_to_index(
                            chunk,
                            source_type='arxiv_papers',
                            document_id=document_id,
                            file_name=pdf_file,
                            chunk_index=i,
                            metadata=paper_metadata
                        )
                    
                    results['arxiv_papers_processed'] += 1
                    results['arxiv_chunks'] += len(chunks)
                    
                    logger.info(f"📄 Processed ArXiv paper: {pdf_file} ({len(chunks)} chunks)")
                
                except Exception as e:
                    error_msg = f"Error processing ArXiv PDF {pdf_file}: {e}"
                    results['processing_errors'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
        
        return results
    
    def _process_web_folder(self, web_folder: str, document_id: str) -> Dict[str, Any]:
        """Process web content from a document folder"""
        results = {
            'web_content_processed': 0,
            'web_chunks': 0,
            'processing_errors': []
        }
        
        # Check for content in content folder
        content_folder = os.path.join(web_folder, 'content')
        if not os.path.exists(content_folder):
            logger.warning(f"⚠️ No content folder found in {web_folder}")
            return results
        
        # Process each JSON content file
        for content_file in os.listdir(content_folder):
            if content_file.endswith('.json'):
                try:
                    content_path = os.path.join(content_folder, content_file)
                    
                    with open(content_path, 'r', encoding='utf-8') as f:
                        content_data = json.load(f)
                    
                    # Extract text content
                    text_content = ""
                    if 'content' in content_data:
                        content_obj = content_data['content']
                        
                        # Combine different text fields
                        if 'full_text' in content_obj:
                            text_content = content_obj['full_text']
                        elif 'summary' in content_obj:
                            text_content = content_obj['summary']
                        elif 'title' in content_obj:
                            text_content = content_obj['title']
                    
                    if not text_content:
                        results['processing_errors'].append(f"No text content found in {content_file}")
                        continue
                    
                    # Create chunks
                    chunks = self._chunk_text(text_content)
                    
                    # Add chunks to Web Content FAISS index
                    for i, chunk in enumerate(chunks):
                        self._add_chunk_to_index(
                            chunk,
                            source_type='web_content',
                            document_id=document_id,
                            file_name=content_file,
                            chunk_index=i,
                            metadata=content_data.get('metadata', {})
                        )
                    
                    results['web_content_processed'] += 1
                    results['web_chunks'] += len(chunks)
                    
                    logger.info(f"🌐 Processed web content: {content_file} ({len(chunks)} chunks)")
                
                except Exception as e:
                    error_msg = f"Error processing web content {content_file}: {e}"
                    results['processing_errors'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
        
        return results
    
    def _add_chunk_to_index(self, chunk: str, source_type: str, document_id: str, 
                           file_name: str, chunk_index: int, metadata: Dict = None):
        """Add a text chunk to appropriate FAISS index with metadata"""
        try:
            # Validate source type
            if source_type not in self.indexes:
                logger.error(f"❌ Invalid source type: {source_type}")
                return
            
            # Create embedding
            embedding = self.embedding_model.encode([chunk])
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding)
            
            # Add to appropriate index
            self.indexes[source_type].add(embedding)
            
            # Store metadata in appropriate collection
            chunk_metadata = {
                'chunk_id': len(self.metadata[source_type]),
                'source_type': source_type,
                'document_id': document_id,
                'file_name': file_name,
                'chunk_index': chunk_index,
                'chunk_text': chunk[:500] + "..." if len(chunk) > 500 else chunk,  # Store truncated text
                'chunk_length': len(chunk),
                'embedding_date': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            self.metadata[source_type].append(chunk_metadata)
            
        except Exception as e:
            logger.error(f"❌ Error adding chunk to {source_type} index: {e}")
    
    def process_user_pdf(self, pdf_path: str, document_id: str = None) -> Dict[str, Any]:
        """
        Process a user uploaded PDF and add to user_pdf index
        
        Args:
            pdf_path: Path to the user uploaded PDF
            document_id: Optional document ID, defaults to filename
            
        Returns:
            Processing results
        """
        if document_id is None:
            document_id = os.path.basename(pdf_path).replace('.pdf', '')
        
        logger.info(f"📄 Processing user PDF: {pdf_path}")
        
        results = {
            'document_id': document_id,
            'pdf_file': os.path.basename(pdf_path),
            'chunks_added': 0,
            'processing_errors': []
        }
        
        try:
            # Extract text from PDF
            pdf_text = self._extract_text_from_pdf(pdf_path)
            if not pdf_text:
                error_msg = f"No text extracted from PDF: {pdf_path}"
                results['processing_errors'].append(error_msg)
                logger.warning(f"⚠️ {error_msg}")
                return results
            
            # Create chunks
            chunks = self._chunk_text(pdf_text)
            
            # Add chunks to User PDF FAISS index
            for i, chunk in enumerate(chunks):
                self._add_chunk_to_index(
                    chunk=chunk,
                    source_type='user_pdf',
                    document_id=document_id,
                    file_name=os.path.basename(pdf_path),
                    chunk_index=i,
                    metadata={'pdf_path': pdf_path, 'upload_date': datetime.now().isoformat()}
                )
            
            results['chunks_added'] = len(chunks)
            
            # Save updated indexes
            self._save_indexes()
            
            logger.info(f"✅ Processed user PDF: {os.path.basename(pdf_path)} ({len(chunks)} chunks)")
            
        except Exception as e:
            error_msg = f"Error processing user PDF {pdf_path}: {e}"
            results['processing_errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return results
    
    def search_similar_content(self, query_text: str, top_k: int = 10, source_types: List[str] = None) -> Dict[str, Any]:
        """
        Search for similar content across organized FAISS indexes
        
        Args:
            query_text: Text to search for
            top_k: Number of similar documents to return per source type
            source_types: List of source types to search in ['user_pdf', 'arxiv_papers', 'web_content']
                         If None, searches all types
            
        Returns:
            Dictionary with results organized by source type
        """
        # Default to searching all source types
        if source_types is None:
            source_types = ['user_pdf', 'arxiv_papers', 'web_content']
        
        # Validate source types
        valid_source_types = [st for st in source_types if st in self.indexes]
        if not valid_source_types:
            logger.warning("⚠️ No valid source types to search")
            return {'results': {}, 'total_results': 0}
        
        try:
            # Create embedding for query
            query_embedding = self.embedding_model.encode([query_text])
            faiss.normalize_L2(query_embedding)
            
            all_results = {}
            total_results = 0
            
            # Search each index separately
            for source_type in valid_source_types:
                if self.indexes[source_type].ntotal == 0:
                    logger.info(f"ℹ️ {source_type} index is empty, skipping...")
                    all_results[source_type] = []
                    continue
                
                # Search in this specific index
                scores, indices = self.indexes[source_type].search(query_embedding, top_k)
                
                source_results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx >= 0 and idx < len(self.metadata[source_type]):  # Valid index
                        result = {
                            'rank': i + 1,
                            'similarity_score': float(score),
                            'chunk_metadata': self.metadata[source_type][idx],
                            'match_quality': self._classify_similarity_score(float(score)),
                            'source_type': source_type
                        }
                        source_results.append(result)
                
                all_results[source_type] = source_results
                total_results += len(source_results)
                
                logger.info(f"🔍 Found {len(source_results)} matches in {source_type}")
            
            return {
                'results': all_results,
                'total_results': total_results,
                'query': query_text,
                'searched_sources': valid_source_types,
                'search_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error searching FAISS indexes: {e}")
            return {'results': {}, 'total_results': 0, 'error': str(e)}
    
    def _classify_similarity_score(self, score: float) -> str:
        """Classify similarity score into quality categories"""
        if score >= 0.85:
            return "Very High"
        elif score >= 0.75:
            return "High"
        elif score >= 0.65:
            return "Medium"
        elif score >= 0.50:
            return "Low"
        else:
            return "Very Low"
    
    def process_all_scraped_data(self, scraped_data_folder: str = None) -> Dict[str, Any]:
        """
        Process all document folders in scraped_data directory
        
        Args:
            scraped_data_folder: Path to scraped_data folder, defaults to Config.SCRAPED_DATA_FOLDER
        """
        if scraped_data_folder is None:
            scraped_data_folder = Config.SCRAPED_DATA_FOLDER
        
        if not os.path.exists(scraped_data_folder):
            logger.error(f"❌ Scraped data folder not found: {scraped_data_folder}")
            return {'error': 'Scraped data folder not found'}
        
        logger.info(f"🔄 Processing all document folders in: {scraped_data_folder}")
        
        overall_results = {
            'total_documents_processed': 0,
            'total_arxiv_papers': 0,
            'total_web_content': 0,
            'total_chunks_added': 0,
            'processing_start': datetime.now().isoformat(),
            'document_results': [],
            'processing_errors': []
        }
        
        # Process each document folder
        for document_folder in os.listdir(scraped_data_folder):
            document_path = os.path.join(scraped_data_folder, document_folder)
            
            if os.path.isdir(document_path):
                try:
                    document_results = self.process_scraped_data_folder(document_path)
                    overall_results['document_results'].append(document_results)
                    
                    # Aggregate results
                    overall_results['total_documents_processed'] += 1
                    overall_results['total_arxiv_papers'] += document_results.get('arxiv_papers_processed', 0)
                    overall_results['total_web_content'] += document_results.get('web_content_processed', 0)
                    overall_results['total_chunks_added'] += document_results.get('total_chunks_added', 0)
                    overall_results['processing_errors'].extend(document_results.get('processing_errors', []))
                    
                except Exception as e:
                    error_msg = f"Error processing document folder {document_folder}: {e}"
                    overall_results['processing_errors'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
        
        overall_results['processing_end'] = datetime.now().isoformat()
        overall_results['faiss_index_stats'] = self.get_index_stats()
        
        logger.info(f"✅ Completed processing: {overall_results['total_documents_processed']} documents, "
                   f"{overall_results['total_chunks_added']} chunks, "
                   f"Total vectors: {overall_results['faiss_index_stats']['total_vectors']}")
        
        return overall_results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the organized FAISS indexes"""
        # Calculate total vectors across all indexes
        total_vectors = sum(index.ntotal for index in self.indexes.values())
        
        # Get unique documents across all source types
        all_documents = set()
        for source_type in self.metadata:
            for chunk in self.metadata[source_type]:
                all_documents.add(chunk['document_id'])
        
        # Get detailed stats for each index
        index_details = {}
        for source_type in ['user_pdf', 'arxiv_papers', 'web_content']:
            index_details[source_type] = {
                'vectors': self.indexes[source_type].ntotal,
                'chunks': len(self.metadata[source_type]),
                'documents': len(set(chunk['document_id'] for chunk in self.metadata[source_type]))
            }
        
        return {
            'total_vectors': total_vectors,
            'dimension': self.embedding_dimension,
            'model_name': self.embedding_model_name,
            'total_documents': len(all_documents),
            'total_chunks': sum(len(metadata) for metadata in self.metadata.values()),
            'index_details': index_details,
            'database_path': self.vector_db_path,
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_source_distribution(self) -> Dict[str, int]:
        """Get distribution of sources across organized indexes"""
        distribution = {}
        for source_type in self.metadata:
            distribution[source_type] = len(self.metadata[source_type])
        return distribution
