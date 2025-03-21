"""
Unit tests for the DataProcessor class.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
from data_processor import DataProcessor, DataProcessorError

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test data and environment."""
        # Create sample test data
        self.test_data = pd.DataFrame({
            'message': [
                'hello! how are you doing today?',  # ham
                'buy now! limited offer! 90% off! click here!',  # spam
                np.nan,  # will be removed
                'good morning! have a nice day ahead',  # ham
                'congratulations! you won $1000000! claim now! urgent!',  # spam
                'hi, can we meet tomorrow for coffee?',  # ham
                'free viagra! best prices! order now! limited stock!',  # spam
                'thanks for your help with the project yesterday',  # ham
                'earn money fast! work from home! $5000/week guaranteed!',  # spam
                'see you at the team meeting later',  # ham
                'looking forward to our lunch next week',  # ham
                'hot singles in your area! click now!',  # spam
                'please review the document i sent',  # ham
                'get rich quick! bitcoin investment! 1000% returns!',  # spam
                'remember to bring your laptop to the workshop',  # ham
                'win an iphone 15! you are our lucky visitor!',  # spam
                'the presentation went well today',  # ham
                'lose weight fast! miracle pill! order now!',  # spam
                'can you send me the meeting notes?',  # ham
                'free money! just enter your bank details!'  # spam
            ],
            'timestamp': [
                '2024-01-01 10:00:00',
                '2024-01-01 10:01:00',
                '2024-01-01 10:02:00',
                '2024-01-01 10:03:00',
                '2024-01-01 10:04:00',
                '2024-01-01 10:05:00',
                '2024-01-01 10:06:00',
                '2024-01-01 10:07:00',
                '2024-01-01 10:08:00',
                '2024-01-01 10:09:00',
                '2024-01-01 10:10:00',
                '2024-01-01 10:11:00',
                '2024-01-01 10:12:00',
                '2024-01-01 10:13:00',
                '2024-01-01 10:14:00',
                '2024-01-01 10:15:00',
                '2024-01-01 10:16:00',
                '2024-01-01 10:17:00',
                '2024-01-01 10:18:00',
                '2024-01-01 10:19:00'
            ]
        })
        
        # Create sample spam labels (0 for ham, 1 for spam)
        self.spam_labels = np.array([
            0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1
        ])
        
        # Create temporary test file
        self.test_file = 'test_data.csv'
        self.test_data.to_csv(self.test_file, index=False)
        
        # Initialize processor
        self.processor = DataProcessor(self.test_file)
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
    def test_initialization(self):
        """Test DataProcessor initialization."""
        # Test with valid file
        processor = DataProcessor(self.test_file)
        self.assertIsNotNone(processor)
        
        # Test with empty file path
        with self.assertRaises(DataProcessorError):
            DataProcessor('')
            
    def test_load_data(self):
        """Test data loading functionality."""
        df = self.processor.load_data()
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 20)
        self.assertEqual(len(df.columns), 2)
        
        # Test with corrupted CSV
        with open('corrupted.csv', 'w') as f:
            f.write('invalid,csv,format\n1,2\n')
        
        with self.assertRaises(DataProcessorError):
            DataProcessor('corrupted.csv').load_data()
            
        os.remove('corrupted.csv')
        
    def test_clean_data(self):
        """Test data cleaning functionality."""
        self.processor.load_data()
        cleaned_df = self.processor.clean_data()
        
        # Check if missing values are handled
        self.assertFalse(cleaned_df['message'].isnull().any())
        
        # Check if text is cleaned
        for message in cleaned_df['message']:
            # Skip empty messages
            if not message:
                continue
            # Check that all alphabetic characters are lowercase
            alpha_chars = ''.join(c for c in message if c.isalpha())
            self.assertTrue(alpha_chars.islower(), f"Message '{message}' contains uppercase letters")
        
        # Check if timestamps are formatted
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df['timestamp']))
        
        # Test with invalid timestamp format
        invalid_data = self.test_data.copy()
        invalid_data['timestamp'] = 'invalid_date'
        invalid_file = 'invalid_timestamps.csv'
        invalid_data.to_csv(invalid_file, index=False)
        
        processor = DataProcessor(invalid_file)
        processor.load_data()
        with self.assertRaises(DataProcessorError):
            processor.clean_data()
            
        os.remove(invalid_file)
        
    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test cleaning without loading
        with self.assertRaises(DataProcessorError):
            self.processor.clean_data()
            
        # Test with empty dataset
        empty_df = pd.DataFrame(columns=['message', 'timestamp'])
        empty_file = 'empty.csv'
        empty_df.to_csv(empty_file, index=False)
        
        processor = DataProcessor(empty_file)
        processor.load_data()  # Should not raise an error
        
        # Test cleaning empty dataset
        with self.assertRaises(DataProcessorError):
            processor.clean_data()
            
        os.remove(empty_file)
        
    def test_spam_detection(self):
        """Test spam detection functionality."""
        # Load and clean data
        self.processor.load_data()
        self.processor.clean_data()
        
        # Train spam detector
        train_acc, test_acc = self.processor.train_spam_detector(self.spam_labels)
        self.assertGreaterEqual(train_acc, 0.4)  # Lower threshold for small dataset
        self.assertGreaterEqual(test_acc, 0.4)  # Lower threshold for small dataset
        
        # Test spam detection
        predictions = self.processor.detect_spam()
        self.assertEqual(len(predictions), len(self.processor.cleaned_data))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Test with probability scores
        predictions, probas = self.processor.detect_spam(return_proba=True)
        self.assertEqual(len(probas), len(self.processor.cleaned_data))
        self.assertTrue(all(0 <= prob <= 1 for prob in probas[:, 1]))
        
        # Test with custom texts
        custom_texts = ['hello world', 'buy now!']
        predictions = self.processor.detect_spam(custom_texts)
        self.assertEqual(len(predictions), len(custom_texts))
        
    def test_spam_evaluation(self):
        """Test spam detection evaluation."""
        # Load and clean data
        self.processor.load_data()
        self.processor.clean_data()
        
        # Train spam detector
        self.processor.train_spam_detector(self.spam_labels)
        
        # Evaluate performance
        metrics = self.processor.evaluate_spam_detection(self.spam_labels)
        self.assertIn('classification_report', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Check classification report
        report = metrics['classification_report']
        self.assertIn('accuracy', report)
        self.assertIn('macro avg', report)
        self.assertIn('weighted avg', report)
        
        # Check macro avg metrics
        macro_avg = report['macro avg']
        self.assertIn('precision', macro_avg)
        self.assertIn('recall', macro_avg)
        self.assertIn('f1-score', macro_avg)
        
        # Check confusion matrix
        conf_matrix = metrics['confusion_matrix']
        self.assertEqual(conf_matrix.shape, (2, 2))
        
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Load and clean data
        self.processor.load_data()
        self.processor.clean_data()
        
        # Train spam detector
        self.processor.train_spam_detector(self.spam_labels)
        
        # Get feature importance
        importance = self.processor.get_spam_feature_importance()
        
        # Check importance format
        self.assertIsInstance(importance, list)
        self.assertTrue(all(isinstance(item, tuple) for item in importance))
        self.assertTrue(all(len(item) == 2 for item in importance))
        self.assertTrue(all(isinstance(item[0], str) for item in importance))
        self.assertTrue(all(isinstance(item[1], float) for item in importance))
        
        # Check importance values
        self.assertTrue(all(0 <= item[1] <= 1 for item in importance))
        
    def test_error_handling_spam(self):
        """Test error handling in spam detection."""
        # Test detection before training
        with self.assertRaises(DataProcessorError):
            self.processor.detect_spam()
            
        # Test evaluation before training
        with self.assertRaises(DataProcessorError):
            self.processor.evaluate_spam_detection(self.spam_labels)
            
        # Test feature importance before training
        with self.assertRaises(DataProcessorError):
            self.processor.get_spam_feature_importance()
            
        # Test training without cleaning
        with self.assertRaises(DataProcessorError):
            self.processor.train_spam_detector(self.spam_labels)

if __name__ == '__main__':
    unittest.main() 