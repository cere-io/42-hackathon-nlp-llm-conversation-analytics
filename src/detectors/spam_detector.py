"""
Spam detection module for identifying spam messages in conversations.

This module provides functionality for:
1. Training a spam detection model
2. Predicting spam messages
3. Evaluating model performance
4. Feature extraction and analysis
"""

import logging
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from text_vectorizer import TextVectorizer

logger = logging.getLogger(__name__)

class SpamDetector:
    """
    A class for detecting spam messages in conversations.
    
    This class provides methods for:
    - Training a spam detection model
    - Predicting spam messages
    - Evaluating model performance
    - Feature analysis
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        n_components: Optional[int] = None,
        cache_ttl: int = 3600,
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42
    ):
        """
        Initialize the SpamDetector.
        
        Args:
            max_features: Maximum number of features to extract from text
            n_components: Number of components for dimensionality reduction
            cache_ttl: Time-to-live for vector cache in seconds
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: Range of n-gram sizes to consider
            random_state: Random seed for reproducibility
        """
        self.vectorizer = TextVectorizer(
            max_features=max_features,
            n_components=n_components,
            cache_ttl=cache_ttl,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )
        
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        
        self.is_fitted = False
        
        logger.info(
            "Initialized SpamDetector with max_features=%d, n_components=%s",
            max_features,
            n_components
        )
    
    def fit(
        self,
        texts: Union[List[str], pd.Series],
        labels: Union[List[int], np.ndarray, pd.Series],
        test_size: float = 0.2
    ) -> Tuple[float, float]:
        """
        Train the spam detection model.
        
        Args:
            texts: List or Series of text messages
            labels: Binary labels (0 for ham, 1 for spam)
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (train_accuracy, test_accuracy)
            
        Raises:
            ValueError: If texts or labels are empty
        """
        # Convert inputs to lists if needed
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(labels, pd.Series):
            labels = labels.values
        if isinstance(labels, list):
            labels = np.array(labels)
            
        # Validate inputs
        if not isinstance(texts, (list, pd.Series)) or len(texts) == 0:
            raise ValueError("Texts cannot be empty")
        if not isinstance(labels, (np.ndarray, pd.Series)) or len(labels) == 0:
            raise ValueError("Labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
            
        logger.info("Training spam detector on %d documents", len(texts))
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # Fit vectorizer and transform texts
        self.vectorizer.fit(X_train)
        X_train_vectors = self.vectorizer.transform(X_train)
        X_test_vectors = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_vectors, y_train)
        self.is_fitted = True
        
        # Calculate accuracies
        train_accuracy = self.classifier.score(X_train_vectors, y_train)
        test_accuracy = self.classifier.score(X_test_vectors, y_test)
        
        # Log performance metrics
        logger.info(
            "Model performance - Train accuracy: %.3f, Test accuracy: %.3f",
            train_accuracy,
            test_accuracy
        )
        
        return train_accuracy, test_accuracy
    
    def predict(
        self,
        texts: Union[str, List[str], pd.Series],
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict spam labels for input texts.
        
        Args:
            texts: Single text or list/Series of texts to predict
            return_proba: Whether to return probability scores
            
        Returns:
            Array of predicted labels (0 for ham, 1 for spam)
            If return_proba is True, also returns probability scores
            
        Raises:
            ValueError: If texts is empty or model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        if len(texts) == 0:
            raise ValueError("Cannot predict on empty texts")
            
        # Transform texts to vectors
        vectors = self.vectorizer.transform(texts)
        
        # Make predictions
        if return_proba:
            return self.classifier.predict(vectors), self.classifier.predict_proba(vectors)
        return self.classifier.predict(vectors)
    
    def evaluate(
        self,
        texts: Union[List[str], pd.Series],
        labels: Union[List[int], np.ndarray, pd.Series]
    ) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            texts: List or Series of text messages
            labels: Binary labels (0 for ham, 1 for spam)
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            ValueError: If texts or labels are empty
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        # Convert inputs to lists if needed
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(labels, pd.Series):
            labels = labels.values
        if isinstance(labels, list):
            labels = np.array(labels)
            
        if len(texts) == 0:
            raise ValueError("Texts cannot be empty")
        if len(labels) == 0:
            raise ValueError("Labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
            
        # Transform texts to vectors
        vectors = self.vectorizer.transform(texts)
        
        # Get predictions
        predictions = self.classifier.predict(vectors)
        
        # Calculate metrics
        report = classification_report(labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(labels, predictions)
        
        # Log evaluation results
        logger.info("Model evaluation results:")
        logger.info("Classification report: %s", report)
        logger.info("Confusion matrix:\n%s", conf_matrix)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """
        Get the importance of each feature for spam detection.
        
        Returns:
            List of (feature_name, importance) tuples sorted by importance
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        # Get feature names
        feature_names = self.vectorizer.get_feature_names()
        
        # Get feature importances
        importances = self.classifier.feature_importances_
        
        # Sort features by importance
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance 