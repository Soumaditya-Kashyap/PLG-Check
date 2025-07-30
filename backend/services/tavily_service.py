import requests
import trafilatura
import logging
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class TavilyService:
    def __init__(self):
        self.api_key = Config.TAVILY_API_KEY
        self.base_url = "https://api.tavily.com/search"
        self.session = requests.Session()
        
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
        """Extract clean text content from URLs using Trafilatura"""
        extracted_content = []
        
        logger.info(f"Extracting content from {len(urls)} URLs...")
        
        for i, url in enumerate(urls, 1):
            try:
                domain = url.split('/')[2] if '/' in url else url[:30]
                logger.info(f"Processing {i}/{len(urls)}: {domain}")
                
                # Download the webpage
                downloaded = trafilatura.fetch_url(url)
                
                if downloaded:
                    # Extract clean text
                    text = trafilatura.extract(downloaded)
                    
                    if text and len(text) > 100:  # Only include substantial content
                        extracted_content.append({
                            'url': url,
                            'content': text,
                            'length': len(text)
                        })
                        logger.info(f"Extracted {len(text):,} characters")
                    else:
                        logger.info("Insufficient content found")
                else:
                    logger.info("Failed to download webpage")
                    
            except Exception as e:
                logger.info(f"Error extracting from {url}: {str(e)}")
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
