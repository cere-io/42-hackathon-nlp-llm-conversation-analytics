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
            'message': ['hello!', 'how are you?', np.nan, 'good morning!'],
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', 
                         '2024-01-01 10:02:00', '2024-01-01 10:03:00'],
            'user_id': [1, 2, np.nan, 1],
            'group_id': [1, 1, 1, 1],
            'irrelevant_column': ['a', 'b', 'c', 'd']
        })
        
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
        
        # Test with invalid file path
        with self.assertRaises(DataProcessorError):
            DataProcessor('')
            
        # Test with non-existent file
        with self.assertRaises(DataProcessorError):
            DataProcessor('non_existent.csv')
            
    def test_load_data(self):
        """Test data loading functionality."""
        df = self.processor.load_data()
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 4)
        self.assertEqual(len(df.columns), 5)
        
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
        
        # Check if irrelevant columns are removed
        self.assertNotIn('irrelevant_column', cleaned_df.columns)
        
        # Check if missing values are handled
        self.assertFalse(cleaned_df['message'].isnull().any())
        self.assertFalse(cleaned_df['user_id'].isnull().any())
        
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
        
    def test_get_basic_stats(self):
        """Test basic statistics calculation."""
        self.processor.load_data()
        self.processor.clean_data()
        stats = self.processor.get_basic_stats()
        
        # Check basic stats
        self.assertEqual(stats['total_messages'], 4)
        self.assertEqual(stats['unique_users'], 2)
        self.assertIsInstance(stats['date_range']['start'], datetime)
        self.assertIsInstance(stats['date_range']['end'], datetime)
        
        # Check additional stats
        self.assertIn('count', stats['messages_per_user'])
        self.assertIn('missing_values', stats)
        
    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test cleaning without loading
        with self.assertRaises(DataProcessorError):
            self.processor.clean_data()
            
        # Test stats without cleaning
        self.processor.load_data()
        with self.assertRaises(DataProcessorError):
            self.processor.get_basic_stats()
            
        # Test with empty dataset
        empty_df = pd.DataFrame(columns=['message', 'timestamp', 'user_id', 'group_id'])
        empty_file = 'empty.csv'
        empty_df.to_csv(empty_file, index=False)
        
        processor = DataProcessor(empty_file)
        processor.load_data()
        with self.assertRaises(DataProcessorError):
            processor.clean_data()
            
        os.remove(empty_file)

    def test_tokenization(self):
        """Test message tokenization functionality."""
        self.processor.load_data()
        self.processor.clean_data()
        
        # Test word tokenization
        word_tokens = self.processor.tokenize_messages('word')
        self.assertIn('tokens', word_tokens.columns)
        self.assertTrue(all(isinstance(tokens, list) for tokens in word_tokens['tokens']))
        
        # Verify word tokenization results
        first_message_tokens = word_tokens['tokens'].iloc[0]
        self.assertEqual(first_message_tokens, ['hello'])
        
        # Test sentence tokenization
        sentence_tokens = self.processor.tokenize_messages('sentence')
        self.assertTrue(all(isinstance(tokens, list) for tokens in sentence_tokens['tokens']))
        
        # Test n-gram tokenization
        ngram_tokens = self.processor.tokenize_messages('ngram')
        self.assertTrue(all(isinstance(tokens, list) for tokens in ngram_tokens['tokens']))
        
        # Test invalid strategy
        with self.assertRaises(DataProcessorError):
            self.processor.tokenize_messages('invalid_strategy')
            
        # Test tokenization without cleaning
        new_processor = DataProcessor(self.test_file)
        new_processor.load_data()
        with self.assertRaises(DataProcessorError):
            new_processor.tokenize_messages()

if __name__ == '__main__':
    unittest.main() 