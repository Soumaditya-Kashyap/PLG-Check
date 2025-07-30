import requests
import xml.etree.ElementTree as ET
import time
import logging
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class ArxivService:
    def __init__(self):
        self.base_url = Config.ARXIV_BASE_URL
        self.rate_limit = Config.ARXIV_RATE_LIMIT
        self.session = requests.Session()
        
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on arXiv using the API"""
        try:
            # Clean and format the query
            query = self._clean_query(query)
            
            # Build API request parameters
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            logger.info(f"Searching arXiv for: {query[:50]}...")
            
            # Make the API request
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_arxiv_response(response.text)
            
            logger.info(f"Found {len(papers)} arXiv papers")
            
            # Rate limiting
            time.sleep(self.rate_limit)
            
            return papers
            
        except requests.RequestException as e:
            logger.info(f"arXiv search failed: {str(e)}")
            return []
        except Exception as e:
            logger.info(f"arXiv search error: {str(e)}")
            return []
    
    def _clean_query(self, query: str) -> str:
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
    
    def get_paper_content(self, paper: Dict[str, Any]) -> str:
        """Get searchable content from a paper (title + abstract)"""
        content_parts = []
        
        if paper.get('title'):
            content_parts.append(paper['title'])
        
        if paper.get('abstract'):
            content_parts.append(paper['abstract'])
        
        return ' '.join(content_parts)
    
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
