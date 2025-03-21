"""
Unit tests for the TextVectorizer class.
"""

import unittest
import numpy as np
import pandas as pd
from text_vectorizer import TextVectorizer

class TestTextVectorizer(unittest.TestCase):
    """Test cases for TextVectorizer class."""
    
    def setUp(self):
        """Set up test data and environment."""
        self.test_texts = [
            "Hello world!",
            "This is a test message.",
            "Another test message here.",
            "Testing vectorization.",
            "More test data for better vectorization.",
            "Adding some variety to the test data.",
            "Need more text samples for testing.",
            "Different words and phrases help.",
            "Expanding the vocabulary range.",
            "Final test message example."
        ]
        self.vectorizer = TextVectorizer(
            max_features=20,
            n_components=10,
            cache_ttl=3600
        )
    
    def test_initialization(self):
        """Test TextVectorizer initialization."""
        self.assertIsNotNone(self.vectorizer.vectorizer)
        self.assertIsNotNone(self.vectorizer.vector_cache)
        self.assertEqual(self.vectorizer.max_features, 20)
        self.assertEqual(self.vectorizer.requested_n_components, 10)
        self.assertIsNone(self.vectorizer.n_components)  # Will be set after fit
    
    def test_fit_transform(self):
        """Test fitting and transforming texts."""
        # Fit the vectorizer
        self.vectorizer.fit(self.test_texts)
        
        # Get actual number of features
        n_features = len(self.vectorizer.get_feature_names())
        expected_components = min(10, n_features)
        
        # Transform texts
        vectors = self.vectorizer.transform(self.test_texts)
        
        # Print debug information
        print("\nVector shape:", vectors.shape)
        print("Vector norms:", np.linalg.norm(vectors, axis=1))
        print("First vector:", vectors[0])
        
        # Check vector dimensions
        self.assertEqual(vectors.shape[0], len(self.test_texts))
        self.assertEqual(vectors.shape[1], expected_components)
        
        # Check that vectors are not zero
        self.assertFalse(np.allclose(vectors, 0))
        
        # Check that vectors are normalized
        norms = np.linalg.norm(vectors, axis=1)
        print("Vector norms:", norms)
        self.assertTrue(np.allclose(norms, 1.0, rtol=1e-5))
    
    def test_single_text_transform(self):
        """Test transforming a single text."""
        self.vectorizer.fit(self.test_texts)
        
        # Get actual number of features
        n_features = len(self.vectorizer.get_feature_names())
        expected_components = min(10, n_features)
        
        # Transform single text
        vector = self.vectorizer.transform("Test message")
        
        # Print debug information
        print("\nSingle vector shape:", vector.shape)
        print("Single vector norm:", np.linalg.norm(vector))
        print("Single vector:", vector[0])
        
        # Check vector dimensions
        self.assertEqual(vector.shape[0], 1)
        self.assertEqual(vector.shape[1], expected_components)
        
        # Check that vector is normalized
        norm = np.linalg.norm(vector)
        print("Single vector norm:", norm)
        self.assertTrue(np.allclose(norm, 1.0, rtol=1e-5))
    
    def test_caching(self):
        """Test vector caching functionality."""
        self.vectorizer.fit(self.test_texts)
        
        # First transform (should compute)
        vectors1 = self.vectorizer.transform(self.test_texts)
        
        # Second transform (should use cache)
        vectors2 = self.vectorizer.transform(self.test_texts)
        
        # Check that results are identical
        np.testing.assert_array_equal(vectors1, vectors2)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between texts."""
        self.vectorizer.fit(self.test_texts)
        
        # Calculate similarity matrix
        similarity_matrix = self.vectorizer.calculate_similarity(self.test_texts)
        
        # Print debug information
        print("\nSimilarity matrix shape:", similarity_matrix.shape)
        print("Diagonal values:", np.diag(similarity_matrix))
        print("First row:", similarity_matrix[0])
        print("Min similarity:", np.min(similarity_matrix))
        print("Max similarity:", np.max(similarity_matrix))
        
        # Check matrix dimensions
        self.assertEqual(similarity_matrix.shape, (len(self.test_texts), len(self.test_texts)))
        
        # Check diagonal is close to 1 (self-similarity)
        diagonal = np.diag(similarity_matrix)
        print("Diagonal values:", diagonal)
        self.assertTrue(np.allclose(diagonal, 1.0, rtol=1e-3))  # More lenient tolerance
        
        # Check symmetry
        self.assertTrue(np.allclose(similarity_matrix, similarity_matrix.T, rtol=1e-3))
        
        # Check similarity values are between -1 and 1
        self.assertTrue(np.all(similarity_matrix >= -1.0))
        self.assertTrue(np.all(similarity_matrix <= 1.0))
    
    def test_feature_names(self):
        """Test getting feature names."""
        self.vectorizer.fit(self.test_texts)
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names()
        
        # Print debug information
        print("\nFeature names:", feature_names)
        print("Number of features:", len(feature_names))
        
        # Check that we have feature names
        self.assertGreater(len(feature_names), 0)
        self.assertIsInstance(feature_names[0], str)
    
    def test_empty_texts(self):
        """Test handling of empty texts."""
        # Test fit with empty texts
        with self.assertRaises(ValueError):
            self.vectorizer.fit([])
        
        # Test transform with empty texts
        self.vectorizer.fit(self.test_texts)
        with self.assertRaises(ValueError):
            self.vectorizer.transform([])
        
        # Test similarity with empty texts
        with self.assertRaises(ValueError):
            self.vectorizer.calculate_similarity([])
    
    def test_unfitted_transform(self):
        """Test transform before fitting."""
        with self.assertRaises(ValueError):
            self.vectorizer.transform("Test message")
    
    def test_unfitted_feature_names(self):
        """Test getting feature names before fitting."""
        with self.assertRaises(ValueError):
            self.vectorizer.get_feature_names()

if __name__ == '__main__':
    unittest.main()