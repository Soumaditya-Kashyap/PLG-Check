import requests
import xml.etree.ElementTree as ET
import time
import logging
import os
import json
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class ArxivService:
    def __init__(self):
        self.base_url = Config.ARXIV_BASE_URL
        self.rate_limit = Config.ARXIV_RATE_LIMIT
        self.session = requests.Session()
        
        # Set up storage directories
        self.storage_base = os.path.join(Config.UPLOAD_FOLDER, 'arxiv_data')
        self.pdfs_dir = os.path.join(self.storage_base, 'pdfs')
        self.metadata_dir = os.path.join(self.storage_base, 'metadata')
        
        # Create directories if they don't exist
        os.makedirs(self.pdfs_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Configure session for better reliability
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_paper_content(self, paper: Dict[str, Any]) -> str:
        """Extract comprehensive content from arXiv paper for better matching"""
        try:
            content_parts = []
            
            # Add title (most important for matching)
            if paper.get('title'):
                content_parts.append(f"Title: {paper['title']}")
            
            # Add abstract (crucial for similarity detection)
            if paper.get('summary'):
                content_parts.append(f"Abstract: {paper['summary']}")
            
            # Add authors for context
            if paper.get('authors'):
                authors_str = ', '.join(paper['authors'][:5])  # Limit to first 5 authors
                content_parts.append(f"Authors: {authors_str}")
            
            # Add categories/subjects
            if paper.get('categories'):
                content_parts.append(f"Categories: {paper['categories']}")
            
            # Combine all parts
            full_content = ' '.join(content_parts)
            
            # Ensure content is substantial
            if len(full_content) > 50:
                return full_content
            else:
                # Fallback to just title and summary
                return f"{paper.get('title', '')} {paper.get('summary', '')}"
                
        except Exception as e:
            logger.error(f"Error extracting paper content: {str(e)}")
            return paper.get('title', '') + ' ' + paper.get('summary', '')

    def _clean_query(self, query: str) -> str:
        """Clean and format query for arXiv search"""
        # Remove special characters and format for arXiv API
        query = query.strip()
        # Replace quotes and special chars that might break arXiv API
        query = query.replace('"', '').replace("'", "")
        # Replace multiple spaces with single space
        query = ' '.join(query.split())
        return query
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on arXiv using the API with improved relevance"""
        try:
            # Clean and format the query
            query = self._clean_query(query)
            
            # Try multiple search strategies for better coverage
            search_strategies = [
                f'all:{query}',  # All fields search
                f'ti:{query}',   # Title search (for exact matches)
                f'abs:{query}',  # Abstract search 
            ]
            
            all_papers = []
            seen_ids = set()
            
            for strategy in search_strategies:
                # Build API request parameters
                params = {
                    'search_query': strategy,
                    'start': 0,
                    'max_results': min(max_results, 15),  # Search more initially
                    'sortBy': 'relevance',
                    'sortOrder': 'descending'
                }
                
                logger.info(f"🔍 Searching arXiv with strategy: {strategy[:50]}...")
                
                try:
                    # Make the API request
                    response = self.session.get(self.base_url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    # Parse XML response
                    papers = self._parse_arxiv_response(response.text)
                    
                    # Add unique papers only
                    for paper in papers:
                        arxiv_id = paper.get('arxiv_id')
                        if arxiv_id and arxiv_id not in seen_ids:
                            seen_ids.add(arxiv_id)
                            all_papers.append(paper)
                    
                    # Rate limiting between strategies
                    time.sleep(self.rate_limit)
                    
                except Exception as e:
                    logger.warning(f"Search strategy '{strategy}' failed: {str(e)}")
                    continue
            
            # Sort by relevance using a scoring function
            scored_papers = self._score_paper_relevance(all_papers, query)
            
            # Return top results
            final_papers = scored_papers[:max_results]
            
            logger.info(f"📋 Found {len(final_papers)} relevant arXiv papers (from {len(all_papers)} total)")
            
            return final_papers
            
        except Exception as e:
            logger.error(f"arXiv search error: {str(e)}")
            return []
    
    def _score_paper_relevance(self, papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Score papers by relevance to the query and sort them"""
        try:
            import re
            from collections import Counter
            
            # Extract key terms from query for scoring
            query_terms = re.findall(r'\b\w{3,}\b', query.lower())
            query_words = set(query_terms)
            
            scored_papers = []
            
            for paper in papers:
                score = 0
                
                # Get paper text for scoring
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()
                
                # Title matches get highest score (exact paper matching)
                title_words = set(re.findall(r'\b\w{3,}\b', title))
                title_matches = len(title_words.intersection(query_words))
                score += title_matches * 10  # High weight for title matches
                
                # Abstract matches
                abstract_words = set(re.findall(r'\b\w{3,}\b', abstract))
                abstract_matches = len(abstract_words.intersection(query_words))
                score += abstract_matches * 3  # Medium weight for abstract matches
                
                # Phrase matching bonus (for finding exact papers)
                for term in query_terms:
                    if len(term) > 4:  # Only for substantial terms
                        if term in title:
                            score += 15  # Very high bonus for title phrase match
                        elif term in abstract:
                            score += 5   # Good bonus for abstract phrase match
                
                # Recent papers get slight bonus (prefer newer research)
                try:
                    published = paper.get('published', '')
                    if published and '2023' in published or '2024' in published:
                        score += 2
                except:
                    pass
                
                # Store score with paper
                paper_with_score = paper.copy()
                paper_with_score['relevance_score'] = score
                scored_papers.append(paper_with_score)
            
            # Sort by score (descending)
            scored_papers.sort(key=lambda p: p.get('relevance_score', 0), reverse=True)
            
            # Log top scores for debugging
            if scored_papers:
                top_3 = scored_papers[:3]
                for i, paper in enumerate(top_3, 1):
                    title_short = paper.get('title', 'No title')[:50]
                    score = paper.get('relevance_score', 0)
                    logger.info(f"🏆 #{i}: Score {score} - {title_short}...")
            
            return scored_papers
            
        except Exception as e:
            logger.error(f"Error scoring papers: {str(e)}")
            # Return papers unsorted if scoring fails
            return papers
        """Clean and prepare query for arXiv API"""
        # Remove special characters and limit length
        query = query.replace('"', '').replace("'", "")
        query = ' '.join(query.split())  # Remove extra whitespace
        
        # Limit query length to avoid API issues
        if len(query) > 200:
            query = query[:200]
        
        return query
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse arXiv API XML response"""
        try:
            papers = []
            root = ET.fromstring(xml_content)
            
            # Define namespace
            namespace = {'atom': 'http://www.w3.org/2005/Atom',
                        'arxiv': 'http://arxiv.org/schemas/atom'}
            
            # Find all entry elements
            entries = root.findall('.//atom:entry', namespace)
            
            for entry in entries:
                paper = {}
                
                # Extract title
                title_elem = entry.find('atom:title', namespace)
                paper['title'] = title_elem.text.strip() if title_elem is not None else ""
                
                # Extract abstract/summary
                summary_elem = entry.find('atom:summary', namespace)
                paper['abstract'] = summary_elem.text.strip() if summary_elem is not None else ""
                
                # Extract authors
                authors = []
                author_elems = entry.findall('atom:author/atom:name', namespace)
                for author_elem in author_elems:
                    authors.append(author_elem.text.strip())
                paper['authors'] = authors
                
                # Extract published date
                published_elem = entry.find('atom:published', namespace)
                paper['published'] = published_elem.text.strip() if published_elem is not None else ""
                
                # Extract arXiv ID
                id_elem = entry.find('atom:id', namespace)
                if id_elem is not None:
                    paper['arxiv_id'] = id_elem.text.split('/')[-1]
                    paper['url'] = id_elem.text
                
                # Extract PDF link
                links = entry.findall('atom:link', namespace)
                for link in links:
                    if link.get('title') == 'pdf':
                        paper['pdf_url'] = link.get('href')
                        break
                
                # Extract categories
                categories = []
                category_elems = entry.findall('atom:category', namespace)
                for cat_elem in category_elems:
                    categories.append(cat_elem.get('term', ''))
                paper['categories'] = categories
                
                # Only include papers with substantial content
                if paper.get('title') and paper.get('abstract'):
                    papers.append(paper)
            
            logger.info(f"Successfully parsed {len(papers)} papers from arXiv response")
            return papers
            
        except ET.ParseError as e:
            logger.error(f"Error parsing arXiv XML response: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing arXiv response: {str(e)}")
            return []
    
    def search_multiple_queries(self, queries: List[str], max_results_per_query: int = 5) -> List[Dict[str, Any]]:
        """Search arXiv with multiple queries to get diverse results"""
        all_papers = []
        seen_ids = set()
        
        for query in queries:
            papers = self.search_papers(query, max_results_per_query)
            
            # Deduplicate based on arXiv ID
            for paper in papers:
                arxiv_id = paper.get('arxiv_id')
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    all_papers.append(paper)
        
        return all_papers

    def download_paper_pdf(self, paper: Dict[str, Any]) -> Optional[str]:
        """
        Download PDF for an arXiv paper and save with organized metadata
        
        Returns:
            Path to downloaded PDF file, or None if download failed
        """
        try:
            arxiv_id = paper.get('arxiv_id')
            pdf_url = paper.get('pdf_url')
            
            if not arxiv_id or not pdf_url:
                logger.warning("Paper missing arXiv ID or PDF URL")
                return None
            
            # Create safe filename
            safe_filename = self._sanitize_filename(f"{arxiv_id}_{paper.get('title', 'untitled')}")
            pdf_path = os.path.join(self.pdfs_dir, f"{safe_filename}.pdf")
            metadata_path = os.path.join(self.metadata_dir, f"{safe_filename}.json")
            
            # Check if already downloaded
            if os.path.exists(pdf_path) and os.path.exists(metadata_path):
                logger.info(f"Paper {arxiv_id} already downloaded")
                return pdf_path
            
            # Download PDF
            logger.info(f"Downloading PDF for {arxiv_id}...")
            response = self.session.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Save PDF
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Create metadata file
            metadata = {
                'arxiv_id': arxiv_id,
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'authors': paper.get('authors', []),
                'published': paper.get('published', ''),
                'categories': paper.get('categories', []),
                'pdf_url': pdf_url,
                'url': paper.get('url', ''),
                'download_date': datetime.now().isoformat(),
                'pdf_path': pdf_path,
                'file_size': os.path.getsize(pdf_path)
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully downloaded {arxiv_id} ({os.path.getsize(pdf_path)} bytes)")
            time.sleep(self.rate_limit)  # Rate limiting
            
            return pdf_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None

    def download_papers_batch(self, papers: List[Dict[str, Any]], max_downloads: int = 5) -> List[Dict[str, Any]]:
        """
        Download multiple papers in batch with organized storage
        
        Returns:
            List of paper metadata with download status
        """
        results = []
        download_count = 0
        
        for paper in papers:
            if download_count >= max_downloads:
                logger.info(f"Reached maximum downloads limit: {max_downloads}")
                break
            
            try:
                pdf_path = self.download_paper_pdf(paper)
                
                result = {
                    'arxiv_id': paper.get('arxiv_id'),
                    'title': paper.get('title', ''),
                    'download_success': pdf_path is not None,
                    'pdf_path': pdf_path,
                    'error': None
                }
                
                if pdf_path:
                    download_count += 1
                    safe_name = self._sanitize_filename(f"{paper.get('arxiv_id')}_{paper.get('title', 'untitled')}")
                    result['metadata_path'] = os.path.join(self.metadata_dir, f"{safe_name}.json")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in batch download for {paper.get('arxiv_id')}: {e}")
                results.append({
                    'arxiv_id': paper.get('arxiv_id'),
                    'title': paper.get('title', ''),
                    'download_success': False,
                    'pdf_path': None,
                    'error': str(e)
                })
        
        logger.info(f"Batch download completed: {download_count} successful downloads")
        return results

    def search_and_download(self, keywords: List[str], max_papers: int = 8, custom_pdf_dir: str = None) -> Dict[str, Any]:
        """
        Search for papers using keywords and download them with organized storage
        
        Args:
            keywords: List of keywords to search for
            max_papers: Maximum number of papers to download (flexible range 5-10)
            custom_pdf_dir: Optional custom directory for PDF downloads
        
        Returns:
            Dictionary with search results and download status
        """
        try:
            # Temporarily change download directory if provided
            original_pdfs_dir = self.pdfs_dir
            original_metadata_dir = self.metadata_dir
            
            if custom_pdf_dir:
                self.pdfs_dir = custom_pdf_dir
                self.metadata_dir = os.path.join(custom_pdf_dir, 'metadata')
                os.makedirs(self.metadata_dir, exist_ok=True)
            
            # Use multiple search strategies with keywords
            all_papers = []
            seen_ids = set()
            
            # Strategy 1: Combined keyword search
            combined_query = ' '.join(keywords[:4])  # Use top 4 keywords
            papers = self.search_papers(combined_query, max_papers + 5)  # Search extra for filtering
            
            for paper in papers:
                arxiv_id = paper.get('arxiv_id')
                if arxiv_id and arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    all_papers.append(paper)
            
            # Strategy 2: Individual important keyword searches
            for keyword in keywords[:3]:  # Top 3 most important keywords
                if len(keyword.strip()) > 3:  # Only substantial keywords
                    papers = self.search_papers(keyword.strip(), 5)
                    for paper in papers:
                        arxiv_id = paper.get('arxiv_id')
                        if arxiv_id and arxiv_id not in seen_ids:
                            seen_ids.add(arxiv_id)
                            all_papers.append(paper)
            
            # Score and filter results
            if all_papers:
                scored_papers = self._score_paper_relevance(all_papers, combined_query)
                
                # Ensure we have a good range (5-10 papers)
                target_count = min(max(5, max_papers), min(10, len(scored_papers)))
                final_papers = scored_papers[:target_count]
                
                logger.info(f"📊 Selected {len(final_papers)} papers from {len(all_papers)} candidates")
                
                # Download papers
                download_results = self.download_papers_batch(final_papers, target_count)
                
                successful_downloads = sum(1 for r in download_results if r['download_success'])
                
                result = {
                    'success': True,
                    'query': combined_query,
                    'keywords_used': keywords[:4],
                    'papers_found': len(all_papers),
                    'papers_selected': len(final_papers),
                    'papers_downloaded': successful_downloads,
                    'results': download_results,
                    'storage_paths': {
                        'pdfs_dir': self.pdfs_dir,
                        'metadata_dir': self.metadata_dir
                    }
                }
                
                logger.info(f"✅ Successfully downloaded {successful_downloads}/{target_count} papers")
                
            else:
                result = {
                    'success': False,
                    'query': combined_query,
                    'keywords_used': keywords,
                    'papers_found': 0,
                    'papers_downloaded': 0,
                    'results': [],
                    'error': 'No papers found matching the keywords'
                }
                
                logger.warning("⚠️ No papers found for the given keywords")
            
            # Restore original directories
            if custom_pdf_dir:
                self.pdfs_dir = original_pdfs_dir
                self.metadata_dir = original_metadata_dir
            
            return result
            
        except Exception as e:
            logger.error(f"Error in search and download: {e}")
            # Restore original directories on error
            if custom_pdf_dir:
                self.pdfs_dir = original_pdfs_dir
                self.metadata_dir = original_metadata_dir
            
            return {
                'success': False,
                'error': str(e),
                'query': ' '.join(keywords) if keywords else '',
                'keywords_used': keywords,
                'papers_found': 0,
                'papers_downloaded': 0,
                'results': []
            }

    def _sanitize_filename(self, filename: str) -> str:
        """Create a safe filename by removing/replacing problematic characters"""
        # Remove or replace problematic characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        # Remove multiple underscores and whitespace
        filename = '_'.join(filename.split())
        
        return filename.strip('_')

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about downloaded papers"""
        try:
            pdf_files = [f for f in os.listdir(self.pdfs_dir) if f.endswith('.pdf')]
            metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('.json')]
            
            total_size = 0
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.pdfs_dir, pdf_file)
                total_size += os.path.getsize(pdf_path)
            
            return {
                'total_papers': len(pdf_files),
                'total_metadata_files': len(metadata_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'storage_paths': {
                    'pdfs': self.pdfs_dir,
                    'metadata': self.metadata_dir
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}
