import re
import string
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some'
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def remove_stop_words(self, text: str) -> str:
        """Remove common stop words"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only include substantial sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate basic similarity score between two texts"""
        # Simple word overlap calculation
        words1 = set(self.preprocess_text(text1).split())
        words2 = set(self.preprocess_text(text2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def find_common_phrases(self, text1: str, text2: str, min_length: int = 3) -> List[str]:
        """Find common phrases between two texts"""
        words1 = self.preprocess_text(text1).split()
        words2 = self.preprocess_text(text2).split()
        
        common_phrases = []
        
        # Find common n-grams
        for n in range(min_length, min(len(words1), len(words2)) + 1):
            for i in range(len(words1) - n + 1):
                phrase = ' '.join(words1[i:i+n])
                text2_str = ' '.join(words2)
                if phrase in text2_str:
                    common_phrases.append(phrase)
        
        return list(set(common_phrases))  # Remove duplicates
