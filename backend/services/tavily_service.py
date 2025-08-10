import requests
import trafilatura
import logging
import os
import json
import hashlib
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class TavilyService:
    def __init__(self):
        self.api_key = Config.TAVILY_API_KEY
        self.base_url = "https://api.tavily.com/search"
        self.session = requests.Session()
        
        # Set up storage directories
        self.storage_base = os.path.join(Config.UPLOAD_FOLDER, 'web_data')
        self.content_dir = os.path.join(self.storage_base, 'content')
        self.metadata_dir = os.path.join(self.storage_base, 'metadata')
        
        # Create directories if they don't exist
        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Configure session with anti-403 headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'no-cache'
        })
        
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in configuration")
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using Tavily API"""
        try:
            logger.info(f"Searching web for: {query[:50]}...")
            logger.info(f"Looking for {max_results} web results")
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                'api_key': self.api_key,
                'query': query,
                'search_depth': 'basic',
                'include_answer': False,
                'include_images': False,
                'include_raw_content': False,
                'max_results': max_results
            }
            
            logger.info("Making API request to Tavily...")
            
            response = self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            logger.info(f"Tavily returned {len(results)} web results")
            if results:
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'No title')[:40]
                    url = result.get('url', 'No URL')
                    domain = url.split('/')[2] if '/' in url else url[:30]
                    logger.info(f"{i}. {title}... from {domain}")
            
            return results
            
        except requests.RequestException as e:
            logger.error(f"Error searching via Tavily: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Tavily search: {str(e)}")
            return []
    
    def extract_content_from_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Extract clean text content from URLs using Trafilatura with storage"""
        extracted_content = []
        
        logger.info(f"Extracting content from {len(urls)} URLs...")
        
        for i, url in enumerate(urls, 1):
            try:
                domain = url.split('/')[2] if '/' in url else url[:30]
                logger.info(f"Processing {i}/{len(urls)}: {domain}")
                
                # Generate unique identifier for this URL
                url_hash = hashlib.md5(url.encode()).hexdigest()
                content_file = os.path.join(self.content_dir, f"{url_hash}.txt")
                metadata_file = os.path.join(self.metadata_dir, f"{url_hash}.json")
                
                # Check if already extracted
                if os.path.exists(content_file) and os.path.exists(metadata_file):
                    logger.info(f"Content already extracted for {domain}")
                    with open(content_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    extracted_content.append({
                        'url': url,
                        'content': content,
                        'length': len(content),
                        'content_file': content_file,
                        'metadata_file': metadata_file,
                        'cached': True
                    })
                    continue
                
                # Download the webpage with enhanced headers
                downloaded = trafilatura.fetch_url(
                    url, 
                    config=trafilatura.settings.use_config()
                )
                
                if downloaded:
                    # Extract clean text
                    text = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        include_formatting=False
                    )
                    
                    if text and len(text) > 100:  # Only include substantial content
                        # Save content to file
                        with open(content_file, 'w', encoding='utf-8') as f:
                            f.write(text)
                        
                        # Save metadata
                        metadata = {
                            'url': url,
                            'domain': urlparse(url).netloc,
                            'extraction_date': datetime.now().isoformat(),
                            'content_length': len(text),
                            'content_file': content_file,
                            'url_hash': url_hash
                        }
                        
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        extracted_content.append({
                            'url': url,
                            'content': text,
                            'length': len(text),
                            'content_file': content_file,
                            'metadata_file': metadata_file,
                            'cached': False
                        })
                        logger.info(f"Extracted and saved {len(text):,} characters")
                    else:
                        logger.info("Insufficient content found")
                else:
                    logger.info("Failed to download webpage")
                    
            except Exception as e:
                logger.warning(f"Error extracting from {url}: {str(e)}")
                continue
        
        logger.info(f"Successfully extracted content from {len(extracted_content)}/{len(urls)} URLs")
        return extracted_content
    
    def search_and_extract(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search web and extract content from results"""
        # Search for web results
        search_results = self.search_web(query, max_results)
        
        if not search_results:
            return []
        
        # Extract URLs from search results
        urls = []
        for result in search_results:
            url = result.get('url')
            if url:
                urls.append(url)
        
        # Extract content from URLs
        extracted_content = self.extract_content_from_urls(urls)
        
        # Combine search metadata with extracted content
        combined_results = []
        for i, result in enumerate(search_results):
            url = result.get('url')
            
            # Find corresponding extracted content
            content_data = next((item for item in extracted_content if item['url'] == url), None)
            
            combined_result = {
                'url': url,
                'title': result.get('title', ''),
                'snippet': result.get('content', ''),
                'score': result.get('score', 0),
                'extracted_content': content_data['content'] if content_data else '',
                'content_length': content_data['length'] if content_data else 0
            }
            
            combined_results.append(combined_result)
        
        return combined_results
    
    def search_multiple_queries(self, queries: List[str], max_results_per_query: int = 3) -> List[Dict[str, Any]]:
        """Search web with multiple queries to get diverse results"""
        all_results = []
        seen_urls = set()
        
        for query in queries:
            results = self.search_and_extract(query, max_results_per_query)
            
            # Deduplicate based on URL
            for result in results:
                url = result.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(result)
        
        return all_results
    
    def search_and_store(self, chunk_text: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Search web using chunk text and store results with organized storage
        
        Returns:
            Dictionary with search results and storage information
        """
        try:
            # Search for web results
            search_results = self.search_web(chunk_text, max_results)
            
            if not search_results:
                return {
                    'success': False,
                    'query': chunk_text[:100],
                    'results_found': 0,
                    'content_extracted': 0,
                    'results': []
                }
            
            # Extract URLs from search results
            urls = [result.get('url') for result in search_results if result.get('url')]
            
            # Extract and store content from URLs
            extracted_content = self.extract_content_from_urls(urls)
            
            # Combine search metadata with extracted content
            combined_results = []
            content_count = 0
            
            for result in search_results:
                url = result.get('url')
                
                # Find corresponding extracted content
                content_data = next((item for item in extracted_content if item['url'] == url), None)
                
                combined_result = {
                    'url': url,
                    'title': result.get('title', ''),
                    'snippet': result.get('content', ''),
                    'score': result.get('score', 0),
                    'content_extracted': content_data is not None,
                    'content_length': content_data['length'] if content_data else 0,
                    'storage_files': {
                        'content_file': content_data.get('content_file') if content_data else None,
                        'metadata_file': content_data.get('metadata_file') if content_data else None
                    },
                    'cached': content_data.get('cached', False) if content_data else False
                }
                
                if content_data:
                    content_count += 1
                
                combined_results.append(combined_result)
            
            return {
                'success': True,
                'query': chunk_text[:100],
                'results_found': len(search_results),
                'content_extracted': content_count,
                'results': combined_results,
                'storage_paths': {
                    'content_dir': self.content_dir,
                    'metadata_dir': self.metadata_dir
                }
            }
            
        except Exception as e:
            logger.error(f"Error in search and store: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': chunk_text[:100],
                'results_found': 0,
                'content_extracted': 0,
                'results': []
            }

    def process_chunks_batch(self, chunks: List[str], max_results_per_chunk: int = 2, custom_content_dir: str = None) -> List[Dict[str, Any]]:
        """
        Process multiple text chunks in batch for web search and storage
        
        Args:
            chunks: List of text chunks to process
            max_results_per_chunk: Maximum results per chunk
            custom_content_dir: Optional custom directory for content storage
        
        Returns:
            List of results for each chunk
        """
        # Temporarily change content directory if provided
        original_content_dir = self.content_dir
        original_metadata_dir = self.metadata_dir
        
        if custom_content_dir:
            self.content_dir = custom_content_dir
            self.metadata_dir = os.path.join(custom_content_dir, 'metadata')
            os.makedirs(self.content_dir, exist_ok=True)
            os.makedirs(self.metadata_dir, exist_ok=True)
        
        try:
            batch_results = []
            
            logger.info(f"Processing {len(chunks)} chunks for web search")
            
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
                
                try:
                    result = self.search_and_store(chunk, max_results_per_chunk)
                    result['chunk_index'] = i
                    result['chunk_length'] = len(chunk)
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    batch_results.append({
                        'success': False,
                        'error': str(e),
                        'chunk_index': i,
                        'chunk_length': len(chunk),
                        'results_found': 0,
                        'content_extracted': 0,
                        'results': []
                    })
            
            total_content = sum(r.get('content_extracted', 0) for r in batch_results)
            total_results = sum(r.get('results_found', 0) for r in batch_results)
            
            logger.info(f"Batch processing completed: {total_results} total results, {total_content} content extracted")
            
            return batch_results
        
        finally:
            # Restore original directories
            if custom_content_dir:
                self.content_dir = original_content_dir
                self.metadata_dir = original_metadata_dir

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored web content"""
        try:
            content_files = [f for f in os.listdir(self.content_dir) if f.endswith('.txt')]
            metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('.json')]
            
            total_size = 0
            total_content_length = 0
            
            for content_file in content_files:
                content_path = os.path.join(self.content_dir, content_file)
                total_size += os.path.getsize(content_path)
                
                # Read content length from file
                try:
                    with open(content_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        total_content_length += len(content)
                except:
                    pass
            
            return {
                'total_content_files': len(content_files),
                'total_metadata_files': len(metadata_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_content_length': total_content_length,
                'storage_paths': {
                    'content': self.content_dir,
                    'metadata': self.metadata_dir
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}
