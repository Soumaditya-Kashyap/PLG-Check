import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import os
from config import Config
from services.arxiv_service import ArxivService
from services.tavily_service import TavilyService
from utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class PlagiarismService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD
        
        # Initialize services
        self.arxiv_service = ArxivService()
        self.tavily_service = TavilyService()
        self.text_processor = TextProcessor()
        
        # Initialize collections
        self.arxiv_collection = self._get_or_create_collection("arxiv_papers")
        self.web_collection = self._get_or_create_collection("web_content")
        self.pdf_collection = self._get_or_create_collection("pdf_documents")
        
    def _get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(collection_name)
        except:
            return self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def _generate_id(self, text: str, url: str = "") -> str:
        """Generate a unique ID for text content using content and URL"""
        import uuid
        # Use both content and URL to create more unique IDs
        combined = f"{url}||{text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def store_pdf_content(self, pdf_chunks: List[str], document_id: str) -> None:
        """Store PDF content chunks in ChromaDB"""
        try:
            embeddings = self.embedding_model.encode(pdf_chunks)
            
            ids = [f"{document_id}_chunk_{i}" for i in range(len(pdf_chunks))]
            metadatas = [
                {
                    "document_id": document_id,
                    "chunk_index": i,
                    "content_type": "pdf",
                    "text_length": len(chunk)
                }
                for i, chunk in enumerate(pdf_chunks)
            ]
            
            self.pdf_collection.add(
                embeddings=embeddings.tolist(),
                documents=pdf_chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored {len(pdf_chunks)} PDF chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error storing PDF content: {str(e)}")
            raise
    
    def store_arxiv_papers(self, papers: List[Dict[str, Any]]) -> None:
        """Store arXiv papers in ChromaDB"""
        try:
            documents = []
            metadatas = []
            ids = []
            seen_ids = set()  # Track IDs in current batch
            
            for paper in papers:
                content = self.arxiv_service.get_paper_content(paper)
                if content:
                    arxiv_url = paper.get('url', '')
                    content_id = self._generate_id(content, arxiv_url)
                    
                    # Skip if we've already seen this ID in current batch
                    if content_id in seen_ids:
                        logger.debug(f"Skipping duplicate arXiv paper in current batch: {content_id}")
                        continue
                    
                    # Check if this content already exists in database
                    try:
                        existing = self.arxiv_collection.get(ids=[content_id])
                        if existing['ids']:  # Content already exists, skip it
                            logger.debug(f"Skipping duplicate arXiv paper with ID: {content_id}")
                            continue
                    except Exception:
                        # ID doesn't exist, we can add it
                        pass
                    
                    documents.append(content)
                    ids.append(content_id)
                    seen_ids.add(content_id)  # Mark as seen
                    metadatas.append({
                        "title": paper.get('title', ''),
                        "authors": ', '.join(paper.get('authors', [])),
                        "arxiv_id": paper.get('arxiv_id', ''),
                        "url": paper.get('url', ''),
                        "published": paper.get('published', ''),
                        "content_type": "arxiv"
                    })
            
            if documents:
                embeddings = self.embedding_model.encode(documents)
                
                self.arxiv_collection.add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Stored {len(documents)} arXiv papers")
            
        except Exception as e:
            logger.error(f"Error storing arXiv papers: {str(e)}")
            raise
    
    def store_web_content(self, web_results: List[Dict[str, Any]]) -> None:
        """Store web content in ChromaDB"""
        try:
            documents = []
            metadatas = []
            ids = []
            seen_ids = set()  # Track IDs in current batch
            
            for result in web_results:
                content = result.get('extracted_content', '')
                url = result.get('url', '')
                if content and len(content) > 100:
                    content_id = self._generate_id(content, url)
                    
                    # Skip if we've already seen this ID in current batch
                    if content_id in seen_ids:
                        logger.debug(f"Skipping duplicate content in current batch: {content_id}")
                        continue
                    
                    # Check if this content already exists in database
                    try:
                        existing = self.web_collection.get(ids=[content_id])
                        if existing['ids']:  # Content already exists, skip it
                            logger.debug(f"Skipping duplicate web content with ID: {content_id}")
                            continue
                    except Exception:
                        # ID doesn't exist, we can add it
                        pass
                    
                    documents.append(content)
                    ids.append(content_id)
                    seen_ids.add(content_id)  # Mark as seen
                    metadatas.append({
                        "url": result.get('url', ''),
                        "title": result.get('title', ''),
                        "snippet": result.get('snippet', ''),
                        "score": result.get('score', 0),
                        "content_length": result.get('content_length', 0),
                        "content_type": "web"
                    })
            
            if documents:
                embeddings = self.embedding_model.encode(documents)
                
                self.web_collection.add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Stored {len(documents)} web content items")
            
        except Exception as e:
            logger.error(f"Error storing web content: {str(e)}")
            raise
    
    def find_similar_content_enhanced(self, query_chunks: List[str], n_results: int = 8) -> Dict[str, Any]:
        """Enhanced similarity detection with better matching for arXiv papers"""
        try:
            all_matches = []
            chunk_similarities = {}  # Track highest similarity per chunk
            
            for chunk_idx, chunk in enumerate(query_chunks):
                chunk_embedding = self.embedding_model.encode([chunk])
                
                # Search in arXiv collection with more results
                arxiv_results = self.arxiv_collection.query(
                    query_embeddings=chunk_embedding.tolist(),
                    n_results=n_results
                )
                
                # Search in web collection
                web_results = self.web_collection.query(
                    query_embeddings=chunk_embedding.tolist(),
                    n_results=n_results
                )
                
                # Process arXiv results with enhanced matching
                chunk_max_similarity = 0
                for i, distance in enumerate(arxiv_results['distances'][0]):
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    # Lower threshold for arXiv to catch more potential matches
                    if similarity >= 0.5:  # More sensitive threshold
                        match_data = {
                            'query_chunk': chunk,
                            'matched_content': arxiv_results['documents'][0][i],
                            'similarity': similarity,
                            'source_type': 'arxiv',
                            'metadata': arxiv_results['metadatas'][0][i],
                            'chunk_index': chunk_idx
                        }
                        
                        # Add source title and URL from metadata
                        metadata = arxiv_results['metadatas'][0][i]
                        match_data['source_title'] = metadata.get('title', 'Unknown arXiv paper')
                        match_data['source_url'] = metadata.get('url', '')
                        
                        all_matches.append(match_data)
                        chunk_max_similarity = max(chunk_max_similarity, similarity)
                
                # Process web results
                for i, distance in enumerate(web_results['distances'][0]):
                    similarity = 1 - distance
                    if similarity >= self.similarity_threshold:
                        match_data = {
                            'query_chunk': chunk,
                            'matched_content': web_results['documents'][0][i],
                            'similarity': similarity,
                            'source_type': 'web',
                            'metadata': web_results['metadatas'][0][i],
                            'chunk_index': chunk_idx
                        }
                        
                        # Add source title and URL from metadata
                        metadata = web_results['metadatas'][0][i]
                        match_data['source_title'] = metadata.get('title', 'Unknown web source')
                        match_data['source_url'] = metadata.get('url', '')
                        
                        all_matches.append(match_data)
                        chunk_max_similarity = max(chunk_max_similarity, similarity)
                
                # Track chunk similarity
                chunk_similarities[chunk_idx] = chunk_max_similarity
            
            # Enhanced plagiarism calculation
            total_chunks = len(query_chunks)
            
            # Count chunks with ANY similarity above threshold
            flagged_chunks = sum(1 for sim in chunk_similarities.values() if sim >= 0.6)
            
            # Calculate weighted plagiarism percentage
            if total_chunks > 0:
                # Base calculation
                base_percentage = (flagged_chunks / total_chunks) * 100
                
                # Bonus for high-similarity matches
                high_similarity_matches = [m for m in all_matches if m['similarity'] >= 0.8]
                if high_similarity_matches:
                    bonus = min(20, len(high_similarity_matches) * 5)  # Up to 20% bonus
                    base_percentage += bonus
                
                plagiarism_percentage = min(100, base_percentage)  # Cap at 100%
            else:
                plagiarism_percentage = 0
            
            return {
                'plagiarism_percentage': round(plagiarism_percentage, 2),
                'total_chunks': total_chunks,
                'flagged_chunks': flagged_chunks,
                'matches': sorted(all_matches, key=lambda x: x['similarity'], reverse=True),
                'arxiv_acknowledgment': "Thank you to arXiv for use of its open access interoperability."
            }
            
        except Exception as e:
            logger.error(f"Error finding similar content: {str(e)}")
            raise

    def find_similar_content(self, query_chunks: List[str], n_results: int = 5) -> Dict[str, Any]:
        """Find similar content across all collections"""
        try:
            all_matches = []
            
            for chunk in query_chunks:
                chunk_embedding = self.embedding_model.encode([chunk])
                
                # Search in arXiv collection
                arxiv_results = self.arxiv_collection.query(
                    query_embeddings=chunk_embedding.tolist(),
                    n_results=n_results
                )
                
                # Search in web collection
                web_results = self.web_collection.query(
                    query_embeddings=chunk_embedding.tolist(),
                    n_results=n_results
                )
                
                # Process results
                for i, distance in enumerate(arxiv_results['distances'][0]):
                    similarity = 1 - distance  # Convert distance to similarity
                    if similarity >= self.similarity_threshold:
                        all_matches.append({
                            'query_chunk': chunk,
                            'matched_content': arxiv_results['documents'][0][i],
                            'similarity': similarity,
                            'source_type': 'arxiv',
                            'metadata': arxiv_results['metadatas'][0][i]
                        })
                
                for i, distance in enumerate(web_results['distances'][0]):
                    similarity = 1 - distance  # Convert distance to similarity
                    if similarity >= self.similarity_threshold:
                        all_matches.append({
                            'query_chunk': chunk,
                            'matched_content': web_results['documents'][0][i],
                            'similarity': similarity,
                            'source_type': 'web',
                            'metadata': web_results['metadatas'][0][i]
                        })
            
            # Calculate overall plagiarism statistics
            total_chunks = len(query_chunks)
            flagged_chunks = len(set(match['query_chunk'] for match in all_matches))
            plagiarism_percentage = (flagged_chunks / total_chunks) * 100 if total_chunks > 0 else 0
            
            return {
                'plagiarism_percentage': round(plagiarism_percentage, 2),
                'total_chunks': total_chunks,
                'flagged_chunks': flagged_chunks,
                'matches': sorted(all_matches, key=lambda x: x['similarity'], reverse=True),
                'arxiv_acknowledgment': "Thank you to arXiv for use of its open access interoperability."
            }
            
        except Exception as e:
            logger.error(f"Error finding similar content: {str(e)}")
            raise
    
    def check_plagiarism(self, pdf_chunks: List[str], document_id: str, document_title: str = None) -> Dict[str, Any]:
        """Complete plagiarism check workflow with title-based optimization"""
        try:
            logger.info(f"Starting plagiarism check for document {document_id}")
            logger.info(f"Analyzing {len(pdf_chunks)} text chunks...")
            
            # Store PDF content
            logger.info("Storing PDF content in database...")
            self.store_pdf_content(pdf_chunks, document_id)
            
            # Generate search queries with title optimization
            logger.info("Generating search queries from PDF content...")
            search_queries = self._generate_search_queries_enhanced(pdf_chunks, document_title)
            logger.info(f"Generated {len(search_queries)} search queries")
            
            # Search arXiv for academic papers with title priority
            logger.info("Searching arXiv for academic papers...")
            arxiv_papers = self.arxiv_service.search_multiple_queries(search_queries, max_results_per_query=5)
            
            if arxiv_papers:
                logger.info(f"Found {len(arxiv_papers)} academic papers from arXiv")
                logger.info("Storing arXiv papers in database...")
                self.store_arxiv_papers(arxiv_papers)
            else:
                logger.warning("No academic papers found on arXiv")
            
            # Search web for additional content
            logger.info("Searching web content...")
            web_results = self.tavily_service.search_multiple_queries(search_queries, max_results_per_query=3)
            
            if web_results:
                logger.info(f"Found {len(web_results)} web sources")
                logger.info("Storing web content in database...")
                self.store_web_content(web_results)
            else:
                logger.warning("No web content found")
            
            # Find similar content with enhanced matching
            logger.info("Performing similarity analysis...")
            logger.info(f"Using similarity threshold: {self.similarity_threshold}")
            
            similarity_results = self.find_similar_content_enhanced(pdf_chunks)
            
            # Add source information
            similarity_results['sources_searched'] = {
                'arxiv_papers': len(arxiv_papers),
                'web_pages': len(web_results)
            }
            
            # Log detailed results
            plagiarism_pct = similarity_results['plagiarism_percentage']
            total_chunks = similarity_results['total_chunks']
            flagged_chunks = similarity_results['flagged_chunks']
            matches = similarity_results.get('matches', [])
            
            logger.info("SIMILARITY ANALYSIS RESULTS:")
            logger.info(f"Overall Plagiarism: {plagiarism_pct}%")
            logger.info(f"Total Chunks: {total_chunks}")
            logger.info(f"Flagged Chunks: {flagged_chunks}")
            logger.info(f"Similar Matches: {len(matches)}")
            
            if matches:
                logger.info("TOP MATCHES found:")
                for i, match in enumerate(matches[:3], 1):
                    similarity = match.get('similarity', 0) * 100
                    source = match.get('source_type', 'unknown')
                    metadata = match.get('metadata', {})
                    title = metadata.get('title', 'No title')[:50]
                    logger.info(f"{i}. {similarity:.1f}% similar - {source}")
                    logger.info(f"   Title: {title}")
            
            logger.info(f"Plagiarism check completed for document {document_id}")
            
            return similarity_results
            
        except Exception as e:
            logger.error(f"Error in plagiarism check for document {document_id}: {str(e)}")
            raise
    
    def _generate_search_queries_enhanced(self, pdf_chunks: List[str], document_title: str = None, max_queries: int = 8) -> List[str]:
        """Generate enhanced search queries with title prioritization"""
        queries = []
        
        # Priority 1: Use document title if available
        if document_title:
            logger.info(f"Using document title for priority search: {document_title}")
            queries.append(document_title)
            
            # Extract key terms from title for additional searches
            title_words = document_title.split()
            if len(title_words) > 3:
                # Create shorter title-based queries
                queries.append(' '.join(title_words[:5]))  # First 5 words
                queries.append(' '.join(title_words[-5:]))  # Last 5 words
        
        # Priority 2: Use chunk content for detailed matching
        for chunk in pdf_chunks[:max_queries]:
            processed_chunk = self.text_processor.preprocess_text(chunk)
            
            # Strategy 1: First meaningful sentence
            sentences = processed_chunk.split('.')
            for sentence in sentences[:2]:  # First 2 sentences
                words = sentence.strip().split()
                if 5 <= len(words) <= 15:  # Good query length
                    queries.append(' '.join(words))
                    break
            
            # Strategy 2: Key phrases (first 10 words)
            words = processed_chunk.split()[:12]  # Slightly longer for better matching
            if len(words) >= 5:
                queries.append(' '.join(words))
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for query in queries:
            if query not in seen:
                unique_queries.append(query)
                seen.add(query)
        
        return unique_queries[:max_queries]

    def _generate_search_queries(self, pdf_chunks: List[str], max_queries: int = 5) -> List[str]:
        """Generate search queries from PDF content"""
        queries = []
        
        # Take the first few substantial chunks as queries
        for chunk in pdf_chunks[:max_queries]:
            # Clean and shorten the chunk for search
            processed_chunk = self.text_processor.preprocess_text(chunk)
            
            # Take first few words as query
            words = processed_chunk.split()[:10]  # First 10 words
            if len(words) >= 3:
                queries.append(' '.join(words))
        
        return queries
