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
        
        # Default storage properties (will be overridden when called with custom dirs)
        self.content_dir = None
        self.metadata_dir = None
        
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
    
    def _structure_content_as_json(self, text: str, url: str, title: str = "") -> Dict[str, Any]:
        """Structure extracted content into a JSON format optimized for FAISS embeddings"""
        import re
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract title if not provided
        if not title:
            # Try to find title from content (first line, all caps, or first sentence)
            lines = text.split('\n')
            potential_titles = []
            
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line and len(line) < 200:
                    # Check for title patterns
                    if (line.isupper() and len(line) > 10) or \
                       (not line.endswith('.') and len(line) > 20) or \
                       any(word in line.lower() for word in ['title:', 'abstract:', 'paper:']):
                        potential_titles.append(line)
            
            title = potential_titles[0] if potential_titles else text[:100] + "..."
        
        # Split into sentences for better embedding
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]  # Keep more sentences
        
        # Extract key sections (introduction, methodology, results, conclusion)
        sections = {}
        
        # Try to identify sections by looking for section headers
        section_keywords = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'background', 'motivation', 'overview'],
            'methodology': ['method', 'approach', 'algorithm', 'technique', 'implementation', 'experimental', 'procedure'],
            'results': ['results', 'findings', 'evaluation', 'experiments', 'performance', 'analysis'],
            'discussion': ['discussion', 'interpretation', 'implications'],
            'conclusion': ['conclusion', 'summary', 'discussion', 'future work', 'conclusions'],
            'references': ['references', 'bibliography', 'citations']
        }
        
        # Split text into paragraphs for better section detection
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:  # Fallback: split by single newlines
            paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 50]
        
        # Try to identify sections based on content structure
        text_lower = text.lower()
        for section_name, keywords in section_keywords.items():
            section_content = []
            
            # Look for section headers and extract following content
            for keyword in keywords:
                if keyword in text_lower:
                    keyword_pos = text_lower.find(keyword)
                    
                    # Find the start and end of this section
                    section_start = keyword_pos
                    
                    # Look for next section or take substantial content
                    section_end = len(text)
                    for next_section, next_keywords in section_keywords.items():
                        if next_section != section_name:
                            for next_keyword in next_keywords:
                                next_pos = text_lower.find(next_keyword, keyword_pos + len(keyword))
                                if next_pos > keyword_pos and next_pos < section_end:
                                    section_end = next_pos
                    
                    # Extract section content (minimum 200 chars, maximum 2000 chars per section)
                    if section_end - section_start > 100:
                        section_text = text[section_start:section_end].strip()
                        if len(section_text) > 200:  # Only include substantial sections
                            section_content.append(section_text[:2000])  # Increased limit
                        break
            
            if section_content:
                sections[section_name] = ' '.join(section_content)
        
        # If no specific sections found, create logical sections from content
        if not sections:
            # Divide content into logical chunks
            chunk_size = len(text) // 4  # Divide into 4 parts
            if chunk_size > 500:
                sections['part_1'] = text[:chunk_size].strip()
                sections['part_2'] = text[chunk_size:chunk_size*2].strip()
                sections['part_3'] = text[chunk_size*2:chunk_size*3].strip()
                sections['part_4'] = text[chunk_size*3:].strip()
            else:
                sections['main_content'] = text
        
        # Determine content quality based on length and structure
        quality_score = 0.5  # Default
        if len(text) > 1000:
            quality_score += 0.2
        if len(sentences) > 10:
            quality_score += 0.1
        if any(keyword in text.lower() for keyword in ['research', 'study', 'analysis', 'method']):
            quality_score += 0.2
        
        quality_score = min(1.0, quality_score)
        content_quality = "high" if quality_score > 0.8 else "medium" if quality_score > 0.5 else "low"
        
        # Determine content type
        domain = urlparse(url).netloc.lower()
        content_type = "academic" if any(d in domain for d in ['arxiv', 'ieee', 'acm', 'springer', 'nature']) else \
                      "technical" if any(d in domain for d in ['github', 'stackoverflow', 'medium']) else \
                      "news" if any(d in domain for d in ['news', 'blog', 'post']) else "article"
        
        # Create structured JSON
        structured_content = {
            "metadata": {
                "url": url,
                "title": title,
                "domain": urlparse(url).netloc,
                "extraction_date": datetime.now().isoformat(),
                "content_quality": content_quality,
                "quality_score": round(quality_score, 2)
            },
            "content": {
                "title": title,
                "summary": text[:500] + "..." if len(text) > 500 else text,  # Increased summary length
                "full_text": text,  # Keep complete text as before
                "sections": sections,
                "word_count": len(text.split()),
                "sentence_count": len(sentences),
                "sentences": sentences,  # Include ALL sentences, not just first 20
                "paragraphs": [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20],  # Add paragraphs
                "character_count": len(text)  # Add character count for comparison
            },
            "embedding_metadata": {
                "content_type": content_type,
                "relevance_score": quality_score,
                "language": "en",  # Default to English, could be detected
                "source_type": "web",
                "processing_version": "2.0",  # Updated version
                "extraction_method": "trafilatura_full"  # Method used
            }
        }
        
        return structured_content

    def _create_search_query(self, text: str) -> str:
        """Create an effective search query from chunk text (max 400 chars)"""
        import re
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is short enough, use it directly
        if len(text) <= 400:
            return text
        
        # Extract key phrases and important terms
        # Look for technical terms, proper nouns, and key concepts
        important_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b(?:algorithm|method|approach|technique|model|framework|system)\b[^.]*',
            r'\b(?:artificial intelligence|machine learning|deep learning|neural network|transformer|BERT|GPT)\b[^.]*',
            r'\b(?:research|study|analysis|implementation|development|optimization)\b[^.]*'
        ]
        
        key_phrases = []
        for pattern in important_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_phrases.extend(matches[:2])  # Limit matches per pattern
        
        # If we found key phrases, use them
        if key_phrases:
            query = ' '.join(key_phrases)
            if len(query) <= 400:
                return query
        
        # Fallback: use first 400 characters
        return text[:400].rsplit(' ', 1)[0]  # Break at word boundary

    def _create_multiple_search_queries(self, text: str, max_queries: int = 3) -> List[str]:
        """Create multiple search queries from chunk text for better coverage"""
        import re
        
        queries = []
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Strategy 1: Extract sentences with technical terms
        sentences = re.split(r'[.!?]+', text)
        tech_sentences = []
        tech_terms = [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'algorithm', 'model', 'framework', 'optimization', 'classification', 'regression',
            'transformer', 'attention', 'BERT', 'GPT', 'CNN', 'RNN', 'LSTM'
        ]
        
        for sentence in sentences[:10]:  # Check first 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 20 and any(term.lower() in sentence.lower() for term in tech_terms):
                if len(sentence) <= 400:
                    tech_sentences.append(sentence)
        
        # Add best technical sentences
        queries.extend(tech_sentences[:2])
        
        # Strategy 2: Extract key phrases and concepts
        key_phrase_query = self._create_search_query(text)
        if key_phrase_query not in queries:
            queries.append(key_phrase_query)
        
        # Strategy 3: First paragraph or section
        first_part = text[:400].rsplit(' ', 1)[0] if len(text) > 400 else text
        if first_part not in queries and len(first_part) > 20:
            queries.append(first_part)
        
        # Limit to max_queries
        return queries[:max_queries]

    def search_web_multiple_queries(self, text: str, max_results_per_query: int = 2) -> List[Dict[str, Any]]:
        """Search web using multiple queries for better coverage"""
        all_results = []
        seen_urls = set()
        
        # Generate multiple search queries
        search_queries = self._create_multiple_search_queries(text)
        
        logger.info(f"Generated {len(search_queries)} search queries from text")
        
        for i, query in enumerate(search_queries, 1):
            logger.info(f"Query {i}/{len(search_queries)}: {query[:80]}...")
            
            results = self.search_web(query, max_results_per_query)
            
            # Deduplicate based on URL
            for result in results:
                url = result.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    result['source_query'] = query[:100]  # Track which query found this
                    all_results.append(result)
        
        logger.info(f"Total unique results from all queries: {len(all_results)}")
        return all_results
        """Create an effective search query from chunk text (max 400 chars)"""
        import re
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is short enough, use it directly
        if len(text) <= 400:
            return text
        
        # Extract key phrases and important terms
        # Look for technical terms, proper nouns, and key concepts
        important_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b(?:algorithm|method|approach|technique|model|framework|system)\b[^.]*',
            r'\b(?:artificial intelligence|machine learning|deep learning|neural network|transformer|BERT|GPT)\b[^.]*',
            r'\b(?:research|study|analysis|implementation|development|optimization)\b[^.]*'
        ]
        
        key_phrases = []
        for pattern in important_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_phrases.extend(matches[:2])  # Limit matches per pattern
        
        # If we found key phrases, use them
        if key_phrases:
            query = ' '.join(key_phrases)
            if len(query) <= 400:
                return query
        
        # Fallback: use first 400 characters
        return text[:400].rsplit(' ', 1)[0]  # Break at word boundary
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using Tavily API with proper query formatting"""
        try:
            # Create effective search query (max 400 chars)
            search_query = self._create_search_query(query)
            
            logger.info(f"Original query length: {len(query)}")
            logger.info(f"Search query length: {len(search_query)}")
            logger.info(f"Search query: {search_query[:100]}...")
            logger.info(f"Looking for {max_results} web results")
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                'api_key': self.api_key,
                'query': search_query,
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
                content_file = os.path.join(self.content_dir, f"{url_hash}.json")  # Changed to .json
                metadata_file = os.path.join(self.metadata_dir, f"{url_hash}.json")
                
                # Check if already extracted
                if os.path.exists(content_file) and os.path.exists(metadata_file):
                    logger.info(f"Content already extracted for {domain}")
                    with open(content_file, 'r', encoding='utf-8') as f:
                        structured_content = json.load(f)
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    extracted_content.append({
                        'url': url,
                        'content': structured_content['content']['full_text'],
                        'structured_content': structured_content,
                        'length': structured_content['content']['word_count'],
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
                    # Extract clean text with maximum content settings
                    text = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        include_formatting=True,  # Keep formatting for better structure
                        include_links=False,
                        favor_precision=False,  # Favor recall over precision to get more content
                        favor_recall=True,  # Get as much content as possible
                        output_format='txt'
                    )
                    
                    # If first extraction doesn't get enough content, try with different settings
                    if not text or len(text) < 500:
                        text = trafilatura.extract(
                            downloaded,
                            include_comments=True,  # Include more content
                            include_tables=True,
                            include_formatting=True,
                            favor_recall=True,
                            prune_xpath=[]  # Don't prune any content
                        )
                    
                    if text and len(text) > 100:  # Only include substantial content
                        # Get page title from downloaded content
                        title = ""
                        try:
                            # Try to extract title from the downloaded HTML
                            import re
                            title_match = re.search(r'<title[^>]*>([^<]+)</title>', downloaded, re.IGNORECASE)
                            if title_match:
                                title = title_match.group(1).strip()
                        except:
                            pass
                        
                        # Structure content into JSON format
                        structured_content = self._structure_content_as_json(text, url, title)
                        
                        # Save structured content as JSON
                        with open(content_file, 'w', encoding='utf-8') as f:
                            json.dump(structured_content, f, indent=2, ensure_ascii=False)
                        
                        # Save simplified metadata
                        metadata = {
                            'url': url,
                            'domain': urlparse(url).netloc,
                            'extraction_date': datetime.now().isoformat(),
                            'content_length': len(text),
                            'word_count': structured_content['content']['word_count'],
                            'content_quality': structured_content['metadata']['content_quality'],
                            'content_file': content_file,
                            'url_hash': url_hash
                        }
                        
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        extracted_content.append({
                            'url': url,
                            'content': text,  # Keep original text for backward compatibility
                            'structured_content': structured_content,  # New structured format
                            'length': len(text),
                            'word_count': structured_content['content']['word_count'],
                            'content_file': content_file,
                            'metadata_file': metadata_file,
                            'cached': False
                        })
                        logger.info(f"Extracted and saved {len(text):,} characters in structured JSON format")
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
    
    def search_and_store(self, chunk_text: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search web using chunk text and store results with organized storage
        Uses multiple search strategies for better coverage
        
        Returns:
            Dictionary with search results and storage information
        """
        try:
            # Use multiple search queries for better coverage
            search_results = self.search_web_multiple_queries(chunk_text, max_results_per_query=2)
            
            # Limit total results
            search_results = search_results[:max_results]
            
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
                    'source_query': result.get('source_query', ''),  # Track which query found this
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
            content_files = [f for f in os.listdir(self.content_dir) if f.endswith('.json')]  # Updated to .json
            metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('.json')]
            
            total_size = 0
            total_content_length = 0
            total_word_count = 0
            
            for content_file in content_files:
                content_path = os.path.join(self.content_dir, content_file)
                total_size += os.path.getsize(content_path)
                
                # Read structured content from JSON file
                try:
                    with open(content_path, 'r', encoding='utf-8') as f:
                        structured_content = json.load(f)
                        total_content_length += len(structured_content['content']['full_text'])
                        total_word_count += structured_content['content']['word_count']
                except:
                    pass
            
            return {
                'total_content_files': len(content_files),
                'total_metadata_files': len(metadata_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_content_length': total_content_length,
                'total_word_count': total_word_count,
                'storage_format': 'structured_json',
                'storage_paths': {
                    'content': self.content_dir,
                    'metadata': self.metadata_dir
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}
