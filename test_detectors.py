"""
Unit tests for conversation detectors with caching.

This module contains comprehensive tests for both GPT-4 and Claude 3.5 Sonnet
conversation detectors, including their caching functionality.
"""

import unittest
import tempfile
import shutil
import os
import json
from datetime import datetime
from pathlib import Path
from gpt4 import GPT4ConversationDetector
from claude35 import Claude35ConversationDetector
from conversation_detector import Message, Label

class TestConversationDetectors(unittest.TestCase):
    """Test suite for conversation detectors with caching."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.gpt4_detector = GPT4ConversationDetector(batch_size=2)
        self.claude_detector = Claude35ConversationDetector(batch_size=2)
        
        # Create test messages
        self.messages = [
            Message(
                id="msg1",
                text="Hello, how are you?",
                timestamp=datetime.now().isoformat(),
                user={"username": "user1"}
            ),
            Message(
                id="msg2",
                text="I'm good, thanks!",
                timestamp=datetime.now().isoformat(),
                user={"username": "user2"}
            ),
            Message(
                id="msg3",
                text="What's the weather like?",
                timestamp=datetime.now().isoformat(),
                user={"username": "user1"}
            )
        ]
        
    def tearDown(self):
        """Clean up test environment after each test."""
        shutil.rmtree(self.temp_dir)
        
    def test_gpt4_detector_initialization(self):
        """Test GPT-4 detector initialization."""
        self.assertIsNotNone(self.gpt4_detector.client)
        self.assertEqual(self.gpt4_detector.batch_size, 2)
        self.assertIsNotNone(self.gpt4_detector.cache)
        
    def test_claude_detector_initialization(self):
        """Test Claude detector initialization."""
        self.assertIsNotNone(self.claude_detector.client)
        self.assertEqual(self.claude_detector.batch_size, 2)
        self.assertIsNotNone(self.claude_detector.cache)
        
    def test_gpt4_detector_caching(self):
        """Test GPT-4 detector caching functionality."""
        # First call should cache the result
        labels1 = self.gpt4_detector.detect(self.messages[:2])
        
        # Second call should use cache
        labels2 = self.gpt4_detector.detect(self.messages[:2])
        
        # Results should be identical
        self.assertEqual(labels1, labels2)
        
        # Check cache stats
        stats = self.gpt4_detector.get_cache_stats()
        self.assertEqual(stats["size"], 1)
        
    def test_claude_detector_caching(self):
        """Test Claude detector caching functionality."""
        # First call should cache the result
        labels1 = self.claude_detector.detect(self.messages[:2])
        
        # Second call should use cache
        labels2 = self.claude_detector.detect(self.messages[:2])
        
        # Results should be identical
        self.assertEqual(labels1, labels2)
        
        # Check cache stats
        stats = self.claude_detector.get_cache_stats()
        self.assertEqual(stats["size"], 1)
        
    def test_gpt4_detector_batch_processing(self):
        """Test GPT-4 detector batch processing."""
        # Process all messages
        labels = self.gpt4_detector.detect(self.messages)
        
        # Should process in batches of 2
        self.assertEqual(len(labels), len(self.messages))
        
        # Check cache stats for multiple batches
        stats = self.gpt4_detector.get_cache_stats()
        self.assertEqual(stats["size"], 2)  # Two batches of 2 messages each
        
    def test_claude_detector_batch_processing(self):
        """Test Claude detector batch processing."""
        # Process all messages
        labels = self.claude_detector.detect(self.messages)
        
        # Should process in batches of 2
        self.assertEqual(len(labels), len(self.messages))
        
        # Check cache stats for multiple batches
        stats = self.claude_detector.get_cache_stats()
        self.assertEqual(stats["size"], 2)  # Two batches of 2 messages each
        
    def test_gpt4_detector_error_handling(self):
        """Test GPT-4 detector error handling."""
        # Test with invalid API key
        with self.assertRaises(ValueError):
            GPT4ConversationDetector(api_key="")
            
        # Test with empty message list
        labels = self.gpt4_detector.detect([])
        self.assertEqual(labels, [])
        
    def test_claude_detector_error_handling(self):
        """Test Claude detector error handling."""
        # Test with invalid API key
        with self.assertRaises(ValueError):
            Claude35ConversationDetector(api_key="")
            
        # Test with empty message list
        labels = self.claude_detector.detect([])
        self.assertEqual(labels, [])
        
    def test_gpt4_detector_cache_cleanup(self):
        """Test GPT-4 detector cache cleanup."""
        # Fill cache with multiple batches
        for i in range(5):
            self.gpt4_detector.detect([self.messages[0]])
            
        # Check cache size
        stats = self.gpt4_detector.get_cache_stats()
        self.assertLessEqual(stats["size"], self.gpt4_detector.cache.max_size)
        
    def test_claude_detector_cache_cleanup(self):
        """Test Claude detector cache cleanup."""
        # Fill cache with multiple batches
        for i in range(5):
            self.claude_detector.detect([self.messages[0]])
            
        # Check cache size
        stats = self.claude_detector.get_cache_stats()
        self.assertLessEqual(stats["size"], self.claude_detector.cache.max_size)
        
if __name__ == '__main__':
    unittest.main() 