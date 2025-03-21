"""
Text vectorization module for converting text data into numerical vectors.

This module provides functionality for:
1. Text vectorization using TF-IDF
2. Optional dimensionality reduction
3. Vector caching for improved performance
4. Similarity calculations between vectors
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from cachetools import TTLCache

logger = logging.getLogger(__name__)

class TextVectorizer:
    """
    A class for converting text data into numerical vectors using TF-IDF.
    
    This class provides methods for:
    - Converting text to TF-IDF vectors
    - Reducing vector dimensionality
    - Caching vectors for improved performance
    - Calculating similarity between vectors
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        n_components: Optional[int] = None,
        cache_ttl: int = 3600,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        """
        Initialize the TextVectorizer.
        
        Args:
            max_features: Maximum number of features to extract from text
            n_components: Number of components for dimensionality reduction (None for no reduction)
            cache_ttl: Time-to-live for vector cache in seconds
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: Range of n-gram sizes to consider
        """
        self.max_features = max_features
        self.requested_n_components = n_components
        self.cache_ttl = cache_ttl
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        # Initialize vectorizer with L2 normalization
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words='english',
            norm='l2'  # Ensure L2 normalization
        )
        
        # Initialize dimensionality reduction if needed
        self.reducer = None
        self.n_components = None
            
        # Initialize cache
        self.vector_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        
        logger.info(
            "Initialized TextVectorizer with max_features=%d, requested_n_components=%s",
            max_features,
            n_components
        )
    
    def fit(self, texts: Union[List[str], pd.Series]) -> None:
        """
        Fit the vectorizer on the input texts.
        
        Args:
            texts: List or Series of text documents to fit on
            
        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Cannot fit on empty texts")
            
        logger.info("Fitting vectorizer on %d documents", len(texts))
        self.vectorizer.fit(texts)
        
        # Get the actual number of features
        n_features = len(self.vectorizer.get_feature_names_out())
        logger.info("Actual number of features: %d", n_features)
        
        # Adjust n_components if needed
        if self.requested_n_components is not None:
            self.n_components = min(self.requested_n_components, n_features)
            logger.info(
                "Adjusted n_components from %d to %d based on available features",
                self.requested_n_components,
                self.n_components
            )
            
            # Initialize and fit dimensionality reduction
            self.reducer = TruncatedSVD(n_components=self.n_components)
            tfidf_vectors = self.vectorizer.transform(texts)
            self.reducer.fit(tfidf_vectors)
            logger.info("Fitted dimensionality reduction")
    
    def transform(
        self,
        texts: Union[str, List[str], pd.Series],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Transform input texts into vectors.
        
        Args:
            texts: Single text or list/Series of texts to transform
            use_cache: Whether to use cached vectors if available
            
        Returns:
            Array of vectors representing the input texts
            
        Raises:
            ValueError: If texts is empty or vectorizer is not fitted
        """
        if not hasattr(self.vectorizer, 'vocabulary_'):
            raise ValueError("Vectorizer must be fitted before transform")
            
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            raise ValueError("Cannot transform empty texts")
            
        # Check cache if enabled
        if use_cache:
            cached_vectors = []
            texts_to_transform = []
            
            for text in texts:
                if text in self.vector_cache:
                    cached_vectors.append(self.vector_cache[text])
                else:
                    texts_to_transform.append(text)
                    
            if not texts_to_transform:
                return np.vstack(cached_vectors)
                
            # Transform uncached texts
            new_vectors = self._transform_uncached(texts_to_transform)
            
            # Cache new vectors
            for text, vector in zip(texts_to_transform, new_vectors):
                self.vector_cache[text] = vector
                
            # Combine cached and new vectors
            if cached_vectors:
                return np.vstack([cached_vectors, new_vectors])
            return new_vectors
            
        return self._transform_uncached(texts)
    
    def _transform_uncached(self, texts: List[str]) -> np.ndarray:
        """Transform texts without using cache."""
        # Convert to TF-IDF vectors
        tfidf_vectors = self.vectorizer.transform(texts)
        
        # Convert to dense array
        vectors = tfidf_vectors.toarray()
        
        # Apply dimensionality reduction if configured
        if self.reducer is not None:
            vectors = self.reducer.transform(vectors)
        
        # Handle zero vectors by replacing them with uniform vectors
        zero_vectors = np.all(vectors == 0, axis=1)
        if np.any(zero_vectors):
            n_features = vectors.shape[1]
            uniform_vector = np.ones(n_features) / np.sqrt(n_features)
            vectors[zero_vectors] = uniform_vector
        
        # Normalize vectors to unit length
        vectors = normalize(vectors, norm='l2', axis=1)
        
        # Ensure array is contiguous
        return np.ascontiguousarray(vectors)
    
    def calculate_similarity(
        self,
        texts1: Union[str, List[str], pd.Series],
        texts2: Optional[Union[str, List[str], pd.Series]] = None
    ) -> np.ndarray:
        """
        Calculate cosine similarity between text vectors.
        
        Args:
            texts1: First set of texts
            texts2: Optional second set of texts. If None, calculates similarity within texts1
            
        Returns:
            Matrix of similarity scores
            
        Raises:
            ValueError: If texts1 is empty
        """
        if isinstance(texts1, str):
            texts1 = [texts1]
            
        if not texts1:
            raise ValueError("Cannot calculate similarity for empty texts")
            
        # Transform texts to vectors
        vectors1 = self.transform(texts1)
        
        if texts2 is None:
            # Calculate similarity within texts1
            similarity = cosine_similarity(vectors1)
        else:
            # Calculate similarity between texts1 and texts2
            if isinstance(texts2, str):
                texts2 = [texts2]
            vectors2 = self.transform(texts2)
            similarity = cosine_similarity(vectors1, vectors2)
        
        # Clip similarity values to [-1, 1] range to handle numerical errors
        return np.clip(similarity, -1.0, 1.0)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (terms) used in the vectorization.
        
        Returns:
            List of feature names
            
        Raises:
            ValueError: If vectorizer is not fitted
        """
        if not hasattr(self.vectorizer, 'vocabulary_'):
            raise ValueError("Vectorizer must be fitted before getting feature names")
        return self.vectorizer.get_feature_names_out().tolist() 