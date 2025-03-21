from text_processor import TextProcessor
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from textblob import TextBlob
import re
import logging
from collections import Counter

class AdvancedProcessor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.logger = logging.getLogger(__name__)
        
        # Spam detection patterns
        self.spam_patterns = [
            r'(?i)(airdrop|giveaway|free tokens?|claim now)',
            r'(?i)(click here|join now|limited time)',
            r'(?i)(earn \d+|win \d+|\$\d+)',
            r'(?i)(bot$|_bot|\.bot)',
            r'@\w+',  # Excessive mentions
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        ]
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyzes text sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing polarity and subjectivity
        """
        try:
            analysis = TextBlob(text)
            return {
                'polarity': analysis.sentiment.polarity,  # -1 to 1
                'subjectivity': analysis.sentiment.subjectivity,  # 0 to 1
                'sentiment': 'positive' if analysis.sentiment.polarity > 0 
                           else 'negative' if analysis.sentiment.polarity < 0 
                           else 'neutral'
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment': 'error',
                'error': str(e)
            }
            
    def detect_topics(self, texts: List[str], n_topics: int = 5) -> List[Dict[str, Any]]:
        """
        Detects main topics using TF-IDF and clustering.
        
        Args:
            texts (List[str]): List of texts to analyze
            n_topics (int): Number of topics to detect
            
        Returns:
            List[Dict[str, Any]]: List of topics with their keywords
        """
        try:
            # Process and clean texts
            cleaned_texts = [self.text_processor.clean_text(text) for text in texts]
            
            # Vectorize texts
            tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
            
            # Clustering with DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=2)
            clusters = clustering.fit_predict(tfidf_matrix.toarray())
            
            # Analyze keywords per cluster
            topics = []
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Noise
                    continue
                    
                cluster_docs = [doc for i, doc in enumerate(cleaned_texts) 
                              if clusters[i] == cluster_id]
                              
                # Get most frequent words in cluster
                words = ' '.join(cluster_docs).split()
                top_words = Counter(words).most_common(5)
                
                topics.append({
                    'topic_id': cluster_id,
                    'keywords': [word for word, _ in top_words],
                    'size': len(cluster_docs)
                })
                
            return topics
            
        except Exception as e:
            self.logger.error(f"Error in topic detection: {str(e)}")
            return []
            
    def extract_keywords(self, text: str, top_n: int = 5) -> List[Dict[str, float]]:
        """
        Extracts keywords using TF-IDF.
        
        Args:
            text (str): Text to analyze
            top_n (int): Number of keywords to extract
            
        Returns:
            List[Dict[str, float]]: List of keywords with their scores
        """
        try:
            # Process text
            cleaned_text = self.text_processor.clean_text(text)
            
            # Vectorize
            tfidf_matrix = self.vectorizer.fit_transform([cleaned_text])
            
            # Get keywords
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by score
            keyword_scores = [(feature_names[i], scores[i]) 
                            for i in scores.argsort()[::-1]
                            if scores[i] > 0][:top_n]
                            
            return [{'keyword': word, 'score': float(score)} 
                    for word, score in keyword_scores]
                    
        except Exception as e:
            self.logger.error(f"Error in keyword extraction: {str(e)}")
            return []
            
    def is_spam(self, text: str) -> Dict[str, Any]:
        """
        Detects if a message is spam based on patterns and features.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, Any]: Spam analysis result
        """
        try:
            # Process text
            cleaned_text = self.text_processor.clean_text(text)
            
            # Count pattern matches
            spam_matches = []
            for pattern in self.spam_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    spam_matches.extend(matches)
                    
            # Additional features
            features = {
                'has_urls': bool(re.search(r'http[s]?://', text)),
                'has_mentions': bool(re.search(r'@\w+', text)),
                'all_caps_ratio': len(re.findall(r'[A-Z]', text)) / len(text) if text else 0,
                'exclamation_count': text.count('!'),
                'spam_pattern_matches': len(spam_matches)
            }
            
            # Calculate spam score
            spam_score = (
                features['has_urls'] * 0.3 +
                features['has_mentions'] * 0.2 +
                features['all_caps_ratio'] * 0.15 +
                (features['exclamation_count'] > 2) * 0.15 +
                (features['spam_pattern_matches'] > 0) * 0.2
            )
            
            return {
                'is_spam': spam_score > 0.5,
                'spam_score': spam_score,
                'features': features,
                'matched_patterns': spam_matches
            }
            
        except Exception as e:
            self.logger.error(f"Error in spam detection: {str(e)}")
            return {
                'is_spam': False,
                'spam_score': 0,
                'features': {},
                'error': str(e)
            }
            
    def process_message(self, text: str) -> Dict[str, Any]:
        """
        Processes a message applying all analyses.
        
        Args:
            text (str): Text to process
            
        Returns:
            Dict[str, Any]: Results of all analyses
        """
        # Basic processing
        basic_processing = self.text_processor.process_text(text)
        
        if not basic_processing['success']:
            return basic_processing
            
        # Advanced analyses
        sentiment = self.analyze_sentiment(text)
        keywords = self.extract_keywords(text)
        spam_analysis = self.is_spam(text)
        
        return {
            **basic_processing,
            'sentiment_analysis': sentiment,
            'keywords': keywords,
            'spam_analysis': spam_analysis
        } 