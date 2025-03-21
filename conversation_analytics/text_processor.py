import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import re
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.logger = logging.getLogger(__name__)
        
    def clean_text(self, text: str) -> str:
        """
        Cleans text by removing special characters and normalizing spaces.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyzes text sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing polarity and subjectivity scores
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0
            }

    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extracts noun phrases from text using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of noun phrases
        """
        try:
            blob = TextBlob(text)
            return list(blob.noun_phrases)
        except Exception as e:
            self.logger.error(f"Error extracting noun phrases: {str(e)}")
            return []
        
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenizes and lemmatizes text.
        
        Args:
            text (str): Text to process
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token, pos='v') for token in tokens 
                 if token.lower() not in self.stop_words]
                 
        return tokens
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Processes text completely including sentiment analysis and noun phrase extraction.
        
        Args:
            text (str): Text to process
            
        Returns:
            Dict[str, Any]: Dictionary containing processed text and metadata
        """
        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
                
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Tokenize and lemmatize
            tokens = self.tokenize_and_lemmatize(cleaned_text)
            
            # Sentiment analysis
            sentiment = self.analyze_sentiment(cleaned_text)
            
            # Noun phrase extraction
            noun_phrases = self.extract_noun_phrases(cleaned_text)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'tokens': tokens,
                'token_count': len(tokens),
                'sentiment': sentiment,
                'noun_phrases': noun_phrases,
                'success': True
            }
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return {
                'original_text': text,
                'cleaned_text': '',
                'tokens': [],
                'token_count': 0,
                'sentiment': {'polarity': 0.0, 'subjectivity': 0.0},
                'noun_phrases': [],
                'success': False,
                'error': str(e)
            } 