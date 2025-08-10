"""
LLM Service for keyword extraction using Google Gemma
Extracts relevant keywords from PDF sections for ArXiv search
"""

import os
import google.generativeai as genai
import logging
from typing import List, Dict, Any
import json

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Configure the API key - check both GOOGLE_API_KEY and GEMINI_API_KEY
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("Neither GOOGLE_API_KEY nor GEMINI_API_KEY found in environment variables")
            self.model = None
            return
            
        genai.configure(api_key=api_key)
        
        # Use Gemma model (free tier)
        try:
            self.model = genai.GenerativeModel('gemma-3-27b-it')  # Using Gemma 3 27B model
            logger.info("LLM Service initialized successfully with Gemma 3 27B IT model")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            self.model = None

    def extract_keywords_from_section(self, section_text: str, section_type: str = "unknown") -> Dict[str, Any]:
        """
        Extract relevant academic keywords from a PDF section for ArXiv search
        
        Args:
            section_text: The text content of the section
            section_type: Type of section (introduction, methodology, results, etc.)
            
        Returns:
            Dictionary with keywords and key_phrases lists
        """
        if not self.model:
            logger.error("LLM model not available")
            return self._fallback_keyword_extraction(section_text, section_type)
        
        try:
            prompt = self._create_keyword_extraction_prompt(section_text, section_type)
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                result = self._parse_json_response(response.text, section_type)
                keywords_count = len(result.get('keywords', []))
                phrases_count = len(result.get('key_phrases', []))
                logger.info(f"Extracted {keywords_count} keywords, {phrases_count} phrases from {section_type}")
                return result
            else:
                logger.warning("Empty response from LLM")
                return self._fallback_keyword_extraction(section_text, section_type)
                
        except Exception as e:
            logger.error(f"Error in LLM keyword extraction: {e}")
            return self._fallback_keyword_extraction(section_text, section_type)

    def _create_keyword_extraction_prompt(self, text: str, section_type: str) -> str:
        """Create an optimized prompt for faster keyword extraction"""
        # Truncate text for faster processing
        truncated_text = text[:1200] if len(text) > 1200 else text
        
        return f"""
Extract 5-8 specific academic keywords and 2-3 key phrases from this {section_type} section for scientific paper search.

Focus on: technical terms, research methods, algorithms, scientific concepts, domain names.

Text: {truncated_text}

Return ONLY this JSON format:
{{"keywords": ["term1", "term2"], "key_phrases": ["phrase 1", "phrase 2"]}}

No explanations. Be precise.
"""

    def _parse_json_response(self, response_text: str, section_type: str) -> Dict[str, Any]:
        """Parse the LLM JSON response to extract keywords and phrases"""
        try:
            # Clean up the response text
            cleaned_text = response_text.strip()
            
            # Remove markdown formatting if present
            cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            cleaned_text = cleaned_text.replace('**', '').replace('*', '')
            
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate and clean the parsed data
                keywords = parsed.get('keywords', [])
                key_phrases = parsed.get('key_phrases', [])
                
                # Clean and filter keywords
                clean_keywords = []
                for kw in keywords:
                    if isinstance(kw, str) and 2 <= len(kw.strip()) <= 40:
                        clean_keywords.append(kw.strip().lower())
                
                # Clean and filter phrases
                clean_phrases = []
                for phrase in key_phrases:
                    if isinstance(phrase, str) and 3 <= len(phrase.strip()) <= 60:
                        clean_phrases.append(phrase.strip().lower())
                
                return {
                    'keywords': clean_keywords[:6],  # Limit to 6 keywords
                    'key_phrases': clean_phrases[:3],  # Limit to 3 phrases
                    'section': section_type
                }
            else:
                # Fallback to comma-separated parsing
                return self._parse_fallback_format(cleaned_text, section_type)
                
        except json.JSONDecodeError:
            logger.warning(f"JSON parsing failed for {section_type}, trying fallback")
            return self._parse_fallback_format(response_text, section_type)
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
            return {'keywords': [], 'key_phrases': [], 'section': section_type, 'error': str(e)}

    def _parse_fallback_format(self, text: str, section_type: str) -> Dict[str, Any]:
        """Fallback parsing for comma-separated keywords"""
        try:
            # Try to extract keywords from comma-separated text
            keywords = [kw.strip().lower() for kw in text.split(',') if kw.strip()]
            keywords = [kw for kw in keywords if 2 <= len(kw) <= 40][:6]
            
            return {
                'keywords': keywords,
                'key_phrases': [],
                'section': section_type
            }
        except Exception:
            return {'keywords': [], 'key_phrases': [], 'section': section_type}
        """Parse the LLM response to extract clean keywords"""
        try:
            # Clean up the response
            keywords_text = response_text.strip()
            
            # Remove any markdown formatting
            keywords_text = keywords_text.replace('**', '').replace('*', '')
            
            # Split by commas and clean each keyword
            keywords = [
                keyword.strip().lower() 
                for keyword in keywords_text.split(',')
                if keyword.strip()
            ]
            
            # Filter out very short or very long keywords
            keywords = [
                kw for kw in keywords 
                if 2 <= len(kw) <= 50 and not kw.isdigit()
            ]
            
            return keywords[:10]  # Limit to max 10 keywords
            
        except Exception as e:
            logger.error(f"Error parsing keywords response: {e}")
            return []

    def _fallback_keyword_extraction(self, text: str, section_type: str = "unknown") -> Dict[str, Any]:
        """Simple fallback keyword extraction without LLM"""
        import re
        
        # Simple approach: extract potential technical terms
        words = re.findall(r'\b[A-Za-z][a-zA-Z\s]{2,30}\b', text)
        
        # Filter for potential keywords
        technical_terms = []
        stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'have', 'been', 'was', 'were', 'can', 'will', 'may', 'also', 'such', 'these', 'those'}
        
        for word in words:
            word = word.strip().lower()
            if (3 <= len(word) <= 25 and 
                word not in stop_words and
                not word.isdigit()):
                technical_terms.append(word)
        
        # Return unique terms, limited
        unique_terms = list(set(technical_terms))[:5]
        
        return {
            'keywords': unique_terms,
            'key_phrases': [],
            'section': section_type
        }

    def extract_keywords_batch(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract keywords from multiple sections in batch with improved efficiency
        
        Args:
            sections: List of section dictionaries with 'text', 'type', and 'id' keys
            
        Returns:
            List of dictionaries with extraction results for each section
        """
        results = []
        
        for i, section in enumerate(sections):
            section_id = section.get('id', f"section_{i}")
            section_text = section.get('text', '')
            section_type = section.get('type', 'unknown')
            
            if section_text.strip() and len(section_text.strip()) > 50:  # Only process substantial sections
                result = self.extract_keywords_from_section(section_text, section_type)
                result['section_id'] = section_id
                results.append(result)
                
                keywords_count = len(result.get('keywords', []))
                phrases_count = len(result.get('key_phrases', []))
                logger.info(f"📝 Section {i+1}/{len(sections)} ({section_type}): {keywords_count}kw, {phrases_count}ph")
            else:
                # Skip very short sections
                results.append({
                    'section_id': section_id,
                    'keywords': [],
                    'key_phrases': [],
                    'section': section_type,
                    'skipped': 'too_short'
                })
                
        logger.info(f"🎯 Batch processing complete: {len(results)} sections processed")
        return results

    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.model:
            return {"status": "unavailable", "reason": "Model not initialized"}
        
        return {
            "status": "available",
            "model": "gemma-3-27b-it",
            "provider": "Google",
            "tier": "free"
        }
