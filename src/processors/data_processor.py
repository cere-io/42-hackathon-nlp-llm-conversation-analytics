"""
Data processing module for conversation analytics.

This module provides functionality for:
1. Loading and cleaning conversation data
2. Basic text analysis
3. Spam detection
4. Data validation and error handling
"""

import logging
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
from datetime import datetime
from text_vectorizer import TextVectorizer
from spam_detector import SpamDetector

logger = logging.getLogger(__name__)

class DataProcessorError(Exception):
    """Custom exception for data processing errors."""
    pass

class DataProcessor:
    """
    A class for processing conversation data.
    
    This class provides methods for:
    - Loading and cleaning data
    - Basic text analysis
    - Spam detection
    - Data validation
    """
    
    def __init__(
        self,
        file_path: str,
        max_features: int = 10000,
        n_components: Optional[int] = None,
        cache_ttl: int = 3600
    ):
        """
        Initialize the DataProcessor.
        
        Args:
            file_path: Path to the CSV file containing conversation data
            max_features: Maximum number of features for text vectorization
            n_components: Number of components for dimensionality reduction
            cache_ttl: Time-to-live for vector cache in seconds
            
        Raises:
            DataProcessorError: If file_path is invalid or file doesn't exist
        """
        if not file_path:
            raise DataProcessorError("File path cannot be empty")
            
        self.file_path = file_path
        self.data = None
        self.cleaned_data = None
        
        # Initialize text vectorizer
        self.vectorizer = TextVectorizer(
            max_features=max_features,
            n_components=n_components,
            cache_ttl=cache_ttl
        )
        
        # Initialize spam detector
        self.spam_detector = SpamDetector(
            max_features=max_features,
            n_components=n_components,
            cache_ttl=cache_ttl
        )
        
        logger.info(
            "Initialized DataProcessor with file_path=%s, max_features=%d",
            file_path,
            max_features
        )
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            DataProcessorError: If file is corrupted or required columns are missing
        """
        try:
            # Load data
            self.data = pd.read_csv(self.file_path)
            
            # Check required columns
            required_columns = {'message', 'timestamp'}
            missing_columns = required_columns - set(self.data.columns)
            if missing_columns:
                raise DataProcessorError(
                    f"Missing required columns: {missing_columns}"
                )
            
            logger.info(
                "Loaded data from %s: %d rows, %d columns",
                self.file_path,
                len(self.data),
                len(self.data.columns)
            )
            
            return self.data
            
        except pd.errors.EmptyDataError:
            # Create empty DataFrame with required columns
            self.data = pd.DataFrame(columns=['message', 'timestamp'])
            return self.data
        except pd.errors.ParserError:
            raise DataProcessorError("Invalid CSV format")
        except FileNotFoundError:
            raise DataProcessorError(f"File not found: {self.file_path}")
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data.
        
        Returns:
            DataFrame containing the cleaned data
            
        Raises:
            DataProcessorError: If data hasn't been loaded or is empty
        """
        if self.data is None:
            raise DataProcessorError("Data must be loaded before cleaning")
            
        if self.data.empty:
            raise DataProcessorError("Cannot clean empty dataset")
            
        logger.info("Starting data cleaning process")
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Log initial state
        logger.info("Initial data shape: %s", df.shape)
        logger.info("Missing values before cleaning:\n%s", df.isnull().sum())
        
        # Clean timestamps
        df = self._clean_timestamps(df)
        
        # Clean text data
        df = self._clean_text_data(df)
        
        # Remove irrelevant columns
        df = self._remove_irrelevant_columns(df)
        
        # Log final state
        logger.info("Final data shape: %s", df.shape)
        logger.info("Missing values after cleaning:\n%s", df.isnull().sum())
        
        self.cleaned_data = df
        return df
    
    def _clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean timestamp data."""
        try:
            # Convert to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Remove future timestamps
            current_time = pd.Timestamp.now()
            df = df[df['timestamp'] <= current_time]
            
            logger.info("Cleaned timestamps: %d rows after filtering", len(df))
            
        except Exception as e:
            raise DataProcessorError(f"Error cleaning timestamps: {str(e)}")
            
        return df
    
    def _clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text data."""
        try:
            # Fill missing values with empty string
            df['message'] = df['message'].fillna('')
            
            # Convert to string and clean
            df['message'] = (df['message']
                           .astype(str)
                           .str.lower()
                           .str.strip()
                           .str.replace(r'[^\w\s.,!?-]', '', regex=True)
                           .str.replace(r'\s+', ' ', regex=True))
            
            # Keep only non-empty messages
            df = df[df['message'].str.len() > 0].copy()
            
            logger.info("Cleaned text data: %d rows after filtering", len(df))
            
        except Exception as e:
            raise DataProcessorError(f"Error cleaning text data: {str(e)}")
            
        return df
    
    def _remove_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove irrelevant columns."""
        required_columns = {'message', 'timestamp'}
        columns_to_keep = list(required_columns) + [
            col for col in df.columns
            if col not in required_columns
            and not col.startswith('_')
        ]
        return df[columns_to_keep]
    
    def train_spam_detector(
        self,
        spam_labels: Union[List[int], np.ndarray, pd.Series],
        test_size: float = 0.2
    ) -> Tuple[float, float]:
        """
        Train the spam detection model.
        
        Args:
            spam_labels: Binary labels (0 for ham, 1 for spam)
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (train_accuracy, test_accuracy)
            
        Raises:
            DataProcessorError: If data hasn't been cleaned
        """
        if self.cleaned_data is None:
            raise DataProcessorError("Data must be cleaned before training")
            
        logger.info("Training spam detector on %d messages", len(self.cleaned_data))
        
        # Convert labels to numpy array
        if isinstance(spam_labels, pd.Series):
            spam_labels = spam_labels.values
        if isinstance(spam_labels, list):
            spam_labels = np.array(spam_labels)
            
        # Ensure we have the same number of labels as messages
        if len(spam_labels) != len(self.data):
            raise DataProcessorError(
                f"Number of labels ({len(spam_labels)}) does not match "
                f"number of messages ({len(self.data)})"
            )
            
        # Get indices of non-empty messages
        valid_indices = self.cleaned_data.index
        
        # Filter labels to match cleaned data
        filtered_labels = spam_labels[valid_indices]
        
        # Train the model
        train_acc, test_acc = self.spam_detector.fit(
            self.cleaned_data['message'],
            filtered_labels,
            test_size=test_size
        )
        
        return train_acc, test_acc
    
    def detect_spam(
        self,
        texts: Optional[Union[str, List[str], pd.Series]] = None,
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect spam messages.
        
        Args:
            texts: Optional texts to analyze. If None, uses cleaned data
            return_proba: Whether to return probability scores
            
        Returns:
            Array of predicted labels (0 for ham, 1 for spam)
            If return_proba is True, also returns probability scores
            
        Raises:
            DataProcessorError: If model hasn't been trained
        """
        if not self.spam_detector.is_fitted:
            raise DataProcessorError("Spam detector must be trained before prediction")
            
        # Use cleaned data if no texts provided
        if texts is None:
            if self.cleaned_data is None:
                raise DataProcessorError("Data must be cleaned before prediction")
            texts = self.cleaned_data['message']
            
        # Make predictions
        return self.spam_detector.predict(texts, return_proba)
    
    def evaluate_spam_detection(
        self,
        spam_labels: Union[List[int], np.ndarray, pd.Series]
    ) -> Dict:
        """
        Evaluate spam detection performance.
        
        Args:
            spam_labels: Binary labels (0 for ham, 1 for spam)
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            DataProcessorError: If data hasn't been cleaned or model hasn't been trained
        """
        if self.cleaned_data is None:
            raise DataProcessorError("Data must be cleaned before evaluation")
            
        if not self.spam_detector.is_fitted:
            raise DataProcessorError("Spam detector must be trained before evaluation")
            
        # Convert labels to numpy array
        if isinstance(spam_labels, pd.Series):
            spam_labels = spam_labels.values
        if isinstance(spam_labels, list):
            spam_labels = np.array(spam_labels)
            
        # Ensure we have the same number of labels as messages
        if len(spam_labels) != len(self.data):
            raise DataProcessorError(
                f"Number of labels ({len(spam_labels)}) does not match "
                f"number of messages ({len(self.data)})"
            )
            
        # Get indices of non-empty messages
        valid_indices = self.cleaned_data.index
        
        # Filter labels to match cleaned data
        filtered_labels = spam_labels[valid_indices]
        
        return self.spam_detector.evaluate(
            self.cleaned_data['message'],
            filtered_labels
        )
    
    def get_spam_feature_importance(self) -> List[Tuple[str, float]]:
        """
        Get the importance of each feature for spam detection.
        
        Returns:
            List of (feature_name, importance) tuples sorted by importance
            
        Raises:
            DataProcessorError: If model hasn't been trained
        """
        if not self.spam_detector.is_fitted:
            raise DataProcessorError("Spam detector must be trained before getting feature importance")
            
        return self.spam_detector.get_feature_importance() 