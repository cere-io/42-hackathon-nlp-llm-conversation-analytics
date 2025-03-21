"""
Unit tests for the SpamDetector class.
"""

import unittest
import numpy as np
import pandas as pd
from spam_detector import SpamDetector

class TestSpamDetector(unittest.TestCase):
    """Test cases for SpamDetector class."""
    
    def setUp(self):
        """Set up test data and environment."""
        # Create sample spam and ham messages
        self.spam_texts = [
            "Buy now! Limited time offer!",
            "You've won a prize! Click here!",
            "Special discount on luxury items!",
            "Congratulations! You're a winner!",
            "Exclusive deal just for you!"
        ]
        
        self.ham_texts = [
            "Hello, how are you?",
            "Can we meet tomorrow?",
            "Thanks for your help!",
            "I'll send you the report soon.",
            "Let's discuss the project."
        ]
        
        # Combine texts and create labels
        self.test_texts = self.spam_texts + self.ham_texts
        self.test_labels = [1] * len(self.spam_texts) + [0] * len(self.ham_texts)
        
        # Initialize detector
        self.detector = SpamDetector(
            max_features=20,
            n_components=10,
            cache_ttl=3600
        )
    
    def test_initialization(self):
        """Test SpamDetector initialization."""
        self.assertIsNotNone(self.detector.vectorizer)
        self.assertIsNotNone(self.detector.classifier)
        self.assertFalse(self.detector.is_fitted)
        self.assertEqual(self.detector.vectorizer.max_features, 20)
        self.assertEqual(self.detector.vectorizer.requested_n_components, 10)
    
    def test_fit(self):
        """Test model training."""
        # Train the model
        train_acc, test_acc = self.detector.fit(self.test_texts, self.test_labels)
        
        # Check that model is fitted
        self.assertTrue(self.detector.is_fitted)
        
        # Check that accuracies are reasonable
        self.assertGreater(train_acc, 0.5)
        self.assertGreaterEqual(test_acc, 0.5)
        self.assertLessEqual(train_acc, 1.0)
        self.assertLessEqual(test_acc, 1.0)
    
    def test_predict(self):
        """Test spam prediction."""
        # Train the model
        self.detector.fit(self.test_texts, self.test_labels)
        
        # Test single text prediction
        prediction = self.detector.predict("Buy now! Special offer!")
        self.assertIn(prediction[0], [0, 1])
        
        # Test multiple texts prediction
        predictions = self.detector.predict(self.test_texts)
        self.assertEqual(len(predictions), len(self.test_texts))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        
        # Test prediction with probabilities
        predictions, probabilities = self.detector.predict(
            self.test_texts, return_proba=True
        )
        self.assertEqual(probabilities.shape[1], 2)  # Two classes
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Train the model
        self.detector.fit(self.test_texts, self.test_labels)
        
        # Evaluate the model
        results = self.detector.evaluate(self.test_texts, self.test_labels)
        
        # Check evaluation results
        self.assertIn('classification_report', results)
        self.assertIn('confusion_matrix', results)
        
        # Check confusion matrix shape
        conf_matrix = results['confusion_matrix']
        self.assertEqual(conf_matrix.shape, (2, 2))
        
        # Check classification report
        report = results['classification_report']
        self.assertIn('0', report)
        self.assertIn('1', report)
        self.assertIn('accuracy', report)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Train the model
        self.detector.fit(self.test_texts, self.test_labels)
        
        # Get feature importance
        importance = self.detector.get_feature_importance()
        
        # Check feature importance
        self.assertGreater(len(importance), 0)
        self.assertTrue(all(isinstance(x[0], str) for x in importance))
        self.assertTrue(all(isinstance(x[1], float) for x in importance))
        self.assertTrue(all(0 <= x[1] <= 1 for x in importance))
        
        # Check sorting
        importances = [x[1] for x in importance]
        self.assertEqual(importances, sorted(importances, reverse=True))
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        # Test empty texts in fit
        with self.assertRaises(ValueError):
            self.detector.fit([], self.test_labels)
        with self.assertRaises(ValueError):
            self.detector.fit(self.test_texts, [])
        
        # Test empty texts in predict
        self.detector.fit(self.test_texts, self.test_labels)
        with self.assertRaises(ValueError):
            self.detector.predict([])
        
        # Test empty texts in evaluate
        with self.assertRaises(ValueError):
            self.detector.evaluate([], self.test_labels)
        with self.assertRaises(ValueError):
            self.detector.evaluate(self.test_texts, [])
    
    def test_unfitted_operations(self):
        """Test operations before fitting."""
        # Test predict before fit
        with self.assertRaises(ValueError):
            self.detector.predict("Test message")
        
        # Test evaluate before fit
        with self.assertRaises(ValueError):
            self.detector.evaluate(self.test_texts, self.test_labels)
        
        # Test feature importance before fit
        with self.assertRaises(ValueError):
            self.detector.get_feature_importance()

if __name__ == '__main__':
    unittest.main() 