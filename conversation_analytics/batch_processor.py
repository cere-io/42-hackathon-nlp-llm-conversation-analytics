from typing import List, Dict, Any, Generator
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

class BatchProcessor:
    def __init__(self, 
                 batch_size: int = 1000, 
                 max_memory_mb: int = 1024,
                 language: str = 'en'):
        """
        Initialize the batch processor with memory and batch size limits.
        
        Args:
            batch_size: Number of messages per batch
            max_memory_mb: Memory limit in MB
            language: Language for text processing
        """
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        try:
            self.stop_words = set(stopwords.words(language))
            self.vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
            # Download required NLTK resources
            import nltk
            nltk.download('punkt')
            nltk.download('stopwords')
        except Exception as e:
            self.logger.error(f"Error loading NLP models: {e}")
            raise
        
    def process_messages(self, messages_df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """
        Process messages in batches to optimize memory usage.
        
        Args:
            messages_df: DataFrame with messages to process
            
        Yields:
            DataFrame with the current processed batch of messages
        """
        total_messages = len(messages_df)
        self.logger.info(f"Starting processing of {total_messages} messages in batches of {self.batch_size}")
        
        for start_idx in range(0, total_messages, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_messages)
            batch = messages_df.iloc[start_idx:end_idx].copy()
            
            # Batch processing
            processed_batch = self._process_batch(batch)
            
            self.logger.debug(f"Processed batch {start_idx//self.batch_size + 1} "
                            f"({start_idx} to {end_idx} messages)")
            
            yield processed_batch
            
    def _process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process an individual batch of messages.
        
        Args:
            batch: DataFrame with the batch of messages
            
        Returns:
            DataFrame with processed messages
        """
        # Basic cleaning
        batch['text'] = batch['text'].fillna('')
        batch['text'] = batch['text'].str.strip()
        batch['timestamp'] = pd.to_datetime(batch['timestamp'])
        
        # Remove empty messages
        batch = batch[batch['text'].str.len() > 0]
        
        # Spam detection
        batch['is_spam'] = batch['text'].apply(self._detect_spam)
        
        # Text processing
        batch['cleaned_text'] = batch['text'].apply(self._clean_text)
        batch['tokens'] = batch['cleaned_text'].apply(self._tokenize)
        
        # Vectorization
        batch['vector'] = self._vectorize_texts(batch['cleaned_text'].tolist())
        
        # Sort by timestamp
        batch = batch.sort_values('timestamp')
        
        return batch
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text and remove stopwords.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        return [token for token in tokens if token not in self.stop_words]
    
    def _detect_spam(self, text: str) -> bool:
        """
        Detect if a message is spam.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if spam, False otherwise
        """
        # Common spam patterns
        spam_patterns = [
            r'buy\s+now',
            r'limited\s+time',
            r'click\s+here',
            r'earn\s+money',
            r'free\s+offer',
            r'urgent\s+action',
            r'guaranteed\s+results'
        ]
        
        text = text.lower()
        return any(re.search(pattern, text) for pattern in spam_patterns)
    
    def _vectorize_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Vectorize a list of texts.
        
        Args:
            texts: List of texts to vectorize
            
        Returns:
            List of vectors
        """
        return self.vectorizer.encode(texts)
    
    def save_batch(self, batch: pd.DataFrame, output_dir: str, batch_number: int):
        """
        Save a processed batch to disk.
        
        Args:
            batch: DataFrame with the batch to save
            output_dir: Output directory
            batch_number: Batch number
        """
        output_path = Path(output_dir) / f"processed_batch_{batch_number}.parquet"
        batch.to_parquet(output_path, index=False)
        self.logger.info(f"Batch {batch_number} saved to {output_path}")
    
    def load_batch(self, batch_path: str) -> pd.DataFrame:
        """
        Load a previously saved batch.
        
        Args:
            batch_path: Path to the batch file
            
        Returns:
            DataFrame with the loaded batch
        """
        return pd.read_parquet(batch_path) 