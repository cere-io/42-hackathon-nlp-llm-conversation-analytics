from typing import List, Dict, Any, Generator, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib
import json
from functools import partial

class BatchProcessor:
    def __init__(self, 
                 batch_size: int = 1000, 
                 max_memory_mb: int = 1024,
                 language: str = 'en',
                 n_workers: int = 4,
                 cache_dir: Optional[str] = None):
        """
        Initialize the batch processor with memory and batch size limits.
        
        Args:
            batch_size: Initial number of messages per batch
            max_memory_mb: Memory limit in MB
            language: Language for text processing
            n_workers: Number of worker threads for parallel processing
            cache_dir: Directory for caching results (None to disable caching)
        """
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.language = language
        self.n_workers = n_workers
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'total_processed': 0,
            'total_time': 0,
            'avg_batch_time': 0,
            'memory_usage': [],
            'batch_sizes': []
        }
        
        # Initialize NLP models
        try:
            self.stop_words = set(stopwords.words(language))
            self.vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
            # Download required NLTK resources
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            self.logger.error(f"Error loading NLP models: {e}")
            raise
            
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, batch_data: pd.DataFrame) -> Path:
        """
        Generate a cache file path based on batch data hash.
        
        Args:
            batch_data: DataFrame with the batch data
            
        Returns:
            Path object for the cache file
        """
        if not self.cache_dir:
            return None
            
        # Create a deterministic hash of the batch content
        content_hash = hashlib.md5(
            pd.util.hash_pandas_object(batch_data).values.tobytes()
        ).hexdigest()
        
        return self.cache_dir / f"batch_cache_{content_hash}.parquet"
        
    def _check_memory_usage(self) -> float:
        """
        Check current memory usage and adjust batch size if needed.
        
        Returns:
            Current memory usage in MB
        """
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics['memory_usage'].append(memory_mb)
        
        # Adjust batch size if memory usage is too high
        if memory_mb > self.max_memory_mb * 0.9:  # 90% threshold
            self.batch_size = int(self.batch_size * 0.8)  # Reduce by 20%
            self.logger.warning(f"Memory usage high ({memory_mb:.1f}MB). Reducing batch size to {self.batch_size}")
        elif memory_mb < self.max_memory_mb * 0.5:  # Below 50% threshold
            self.batch_size = min(int(self.batch_size * 1.2), 5000)  # Increase by 20%, max 5000
            
        return memory_mb
        
    def process_messages(self, messages_df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """
        Process messages in batches to optimize memory usage.
        
        Args:
            messages_df: DataFrame with messages to process
            
        Yields:
            DataFrame with the current processed batch of messages
        """
        total_messages = len(messages_df)
        start_time = datetime.now()
        
        self.logger.info(f"Starting processing of {total_messages} messages in batches of {self.batch_size}")
        
        with tqdm(total=total_messages, desc="Processing messages") as pbar:
            for start_idx in range(0, total_messages, self.batch_size):
                # Check and adjust memory usage
                self._check_memory_usage()
                
                end_idx = min(start_idx + self.batch_size, total_messages)
                batch = messages_df.iloc[start_idx:end_idx].copy()
                
                # Check cache first
                cache_path = self._get_cache_path(batch)
                if cache_path and cache_path.exists():
                    processed_batch = pd.read_parquet(cache_path)
                    self.logger.debug(f"Loaded batch from cache: {cache_path}")
                else:
                    # Batch processing
                    batch_start_time = datetime.now()
                    processed_batch = self._process_batch(batch)
                    
                    # Update metrics
                    batch_time = (datetime.now() - batch_start_time).total_seconds()
                    self.metrics['avg_batch_time'] = (
                        (self.metrics['avg_batch_time'] * self.metrics['total_processed'] + batch_time) /
                        (self.metrics['total_processed'] + 1)
                    )
                    
                    # Save to cache if enabled
                    if cache_path:
                        processed_batch.to_parquet(cache_path, index=False)
                
                self.metrics['total_processed'] += len(processed_batch)
                self.metrics['batch_sizes'].append(len(processed_batch))
                
                pbar.update(len(batch))
                self.logger.debug(f"Processed batch {start_idx//self.batch_size + 1} "
                                f"({start_idx} to {end_idx} messages)")
                
                yield processed_batch
                
        # Update final metrics
        self.metrics['total_time'] = (datetime.now() - start_time).total_seconds()
        self._log_performance_summary()
            
    def _process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process an individual batch of messages using parallel processing.
        
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
        
        # Parallel processing of text operations
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Spam detection
            spam_future = executor.submit(
                lambda: batch['text'].apply(self._detect_spam)
            )
            
            # Text cleaning
            clean_future = executor.submit(
                lambda: batch['text'].apply(self._clean_text)
            )
            
            # Wait for results
            batch['is_spam'] = spam_future.result()
            batch['cleaned_text'] = clean_future.result()
            
            # Tokenization
            batch['tokens'] = list(executor.map(
                self._tokenize,
                batch['cleaned_text']
            ))
            
            # Vectorization (done in batches for efficiency)
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
    
    def _log_performance_summary(self):
        """Log a summary of processing performance metrics."""
        summary = {
            'Total messages processed': self.metrics['total_processed'],
            'Total processing time': f"{self.metrics['total_time']:.2f}s",
            'Average time per batch': f"{self.metrics['avg_batch_time']:.2f}s",
            'Average messages per second': f"{self.metrics['total_processed']/self.metrics['total_time']:.1f}",
            'Peak memory usage': f"{max(self.metrics['memory_usage']):.1f}MB",
            'Average batch size': f"{sum(self.metrics['batch_sizes'])/len(self.metrics['batch_sizes']):.1f}"
        }
        
        self.logger.info("Performance Summary:")
        for metric, value in summary.items():
            self.logger.info(f"  {metric}: {value}")
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.metrics.copy() 