"""
Data processing module for conversation analytics.
Handles data loading, cleaning, and preprocessing operations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set
import logging
from datetime import datetime
import re
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessorError(Exception):
    """Base exception class for DataProcessor errors."""
    pass

class DataProcessor:
    """
    Handles data processing operations for conversation analytics.
    
    This class provides functionality for:
    - Loading and validating conversation data
    - Cleaning and preprocessing text
    - Handling missing values
    - Computing basic statistics
    
    Attributes:
        file_path (str): Path to the CSV file containing conversation data
        required_columns (Set[str]): Set of required columns in the dataset
        data (Optional[pd.DataFrame]): Raw data loaded from CSV
        cleaned_data (Optional[pd.DataFrame]): Processed and cleaned data
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the DataProcessor.
        
        Args:
            file_path: Path to the CSV file containing conversation data
            
        Raises:
            DataProcessorError: If the file path is invalid or file doesn't exist
        """
        if not file_path or not isinstance(file_path, str):
            raise DataProcessorError("Invalid file path")
            
        file_path = Path(file_path)
        if not file_path.exists():
            raise DataProcessorError(f"File not found: {file_path}")
            
        self.file_path = str(file_path)
        self.required_columns = {'message', 'timestamp', 'user_id', 'group_id'}
        self.data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and validate conversation data from CSV.
        
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            DataProcessorError: If file is corrupted or missing required columns
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            
            # Validate required columns
            missing_cols = self.required_columns - set(self.data.columns)
            if missing_cols:
                raise DataProcessorError(f"Missing required columns: {missing_cols}")
                
            logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
            return self.data
            
        except pd.errors.EmptyDataError:
            raise DataProcessorError("CSV file is empty")
        except pd.errors.ParserError:
            raise DataProcessorError("CSV file is corrupted or has invalid format")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise DataProcessorError(f"Failed to load data: {str(e)}")
            
    def clean_data(self) -> pd.DataFrame:
        """
        Perform data cleaning and preprocessing operations.
        
        Returns:
            Cleaned DataFrame
            
        Raises:
            DataProcessorError: If cleaning operations fail
        """
        if self.data is None:
            raise DataProcessorError("Data not loaded. Call load_data() first.")
            
        try:
            logger.info("Starting data cleaning process")
            
            # Create a copy to avoid modifying original data
            df = self.data.copy()
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Format timestamps
            df = self._format_timestamps(df)
            
            # Clean text data
            df = self._clean_text_data(df)
            
            # Remove irrelevant columns
            df = self._remove_irrelevant_columns(df)
            
            # Validate cleaned data
            if df.empty:
                raise DataProcessorError("Cleaning resulted in empty dataset")
                
            self.cleaned_data = df
            logger.info("Data cleaning completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            raise DataProcessorError(f"Failed to clean data: {str(e)}")
            
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # Log missing values before cleaning
        missing_counts = df.isnull().sum()
        logger.info(f"Missing values before cleaning:\n{missing_counts[missing_counts > 0]}")
        
        # Fill missing values based on column type
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].fillna('')
            elif df[column].dtype in ['int64', 'float64']:
                df[column] = df[column].fillna(0)
                
        return df
        
    def _format_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format timestamp columns ensuring valid datetime values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with formatted timestamps
            
        Raises:
            DataProcessorError: If timestamp conversion fails
        """
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
            return df
        except ValueError as e:
            raise DataProcessorError(f"Invalid timestamp format: {str(e)}")
        
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned and normalized text string
        """
        # Handle empty or NaN values
        if pd.isna(text) or not str(text).strip():
            return ''
            
        # Convert to string and normalize
        text = str(text)
        
        # Convert to lowercase first
        text = text.lower()
        
        # Remove special characters except allowed punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Final strip and lowercase check
        text = text.strip()
        
        # Verify the result is lowercase (for debugging)
        alpha_only = ''.join(c for c in text if c.isalpha())
        if alpha_only and not alpha_only.islower():
            logger.warning(f"Text still contains uppercase after cleaning: '{text}'")
            text = text.lower()
        
        return text
        
    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize text data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned text
        """
        # Clean message column
        df = df.copy()
        df['message'] = df['message'].apply(self._clean_text)
        
        # Log cleaned messages
        logger.info("Cleaned messages:")
        logger.info(df['message'].tolist())
        
        return df
        
    def _remove_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove non-essential columns from the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with only relevant columns
        """
        return df[list(self.required_columns)]
        
    def get_basic_stats(self) -> Dict:
        """
        Calculate basic statistics about the dataset.
        
        Returns:
            Dictionary containing basic statistics
            
        Raises:
            DataProcessorError: If statistics calculation fails
        """
        if self.cleaned_data is None:
            raise DataProcessorError("No cleaned data available. Run clean_data() first.")
            
        try:
            # Ensure user_id is numeric for counting
            df = self.cleaned_data.copy()
            df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
            
            # Count unique users excluding NaN values and 0 values
            unique_users = df[df['user_id'] > 0]['user_id'].nunique()
            
            stats = {
                'total_messages': len(df),
                'unique_users': unique_users,
                'date_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                },
                'messages_per_user': df.groupby('user_id').size().describe().to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            return stats
            
        except Exception as e:
            raise DataProcessorError(f"Failed to calculate statistics: {str(e)}")

    def tokenize_messages(self, strategy: str = 'word') -> pd.DataFrame:
        """
        Tokenize messages using the specified strategy.
        
        Args:
            strategy: Tokenization strategy to use. Options:
                     - 'word': Simple word-based tokenization
                     - 'sentence': Sentence-based tokenization
                     - 'ngram': N-gram based tokenization (default n=2)
        
        Returns:
            DataFrame with original messages and their tokens
            
        Raises:
            DataProcessorError: If tokenization fails or invalid strategy
        """
        if self.cleaned_data is None:
            raise DataProcessorError("No cleaned data available. Run clean_data() first.")
            
        try:
            df = self.cleaned_data.copy()
            
            if strategy == 'word':
                df['tokens'] = df['message'].apply(self._word_tokenize)
            elif strategy == 'sentence':
                df['tokens'] = df['message'].apply(self._sentence_tokenize)
            elif strategy == 'ngram':
                df['tokens'] = df['message'].apply(lambda x: self._ngram_tokenize(x, n=2))
            else:
                raise DataProcessorError(f"Invalid tokenization strategy: {strategy}")
                
            # Log tokenization results
            logger.info(f"Tokenization completed using strategy: {strategy}")
            logger.info(f"Average tokens per message: {df['tokens'].apply(len).mean():.2f}")
            
            return df
            
        except Exception as e:
            raise DataProcessorError(f"Tokenization failed: {str(e)}")
            
    def _word_tokenize(self, text: str) -> List[str]:
        """
        Perform word-based tokenization.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of word tokens
        """
        if not text:
            return []
            
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', '', text)
        tokens = [token.strip() for token in text.split()]
        return [token for token in tokens if token]
        
    def _sentence_tokenize(self, text: str) -> List[str]:
        """
        Perform sentence-based tokenization.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentence tokens
        """
        if not text:
            return []
            
        # Split on common sentence endings and remove empty sentences
        sentences = re.split(r'[.!?]+', text)
        return [sent.strip() for sent in sentences if sent.strip()]
        
    def _ngram_tokenize(self, text: str, n: int = 2) -> List[str]:
        """
        Perform n-gram based tokenization.
        
        Args:
            text: Input text to tokenize
            n: Size of n-grams (default: 2)
            
        Returns:
            List of n-gram tokens
        """
        if not text:
            return []
            
        words = self._word_tokenize(text)
        if len(words) < n:
            return words
            
        # Generate n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            ngrams.append(ngram)
            
        return ngrams 