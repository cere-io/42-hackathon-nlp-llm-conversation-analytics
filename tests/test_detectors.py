"""
Unit tests for conversation detectors with caching.

This module contains comprehensive tests for both GPT-4 and Claude 3.5 Sonnet
conversation detectors, including their caching functionality.
"""

import unittest
import tempfile
import shutil
import os
from unittest.mock import MagicMock, patch
from datetime import datetime
from gpt4 import GPT4ConversationDetector
from claude35 import Claude35ConversationDetector
from conversation_detector import Message, Label
import json

class TestConversationDetectors(unittest.TestCase):
    """Test suite for conversation detectors with caching."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Test messages
        self.messages = [
            Message(
                content=f"Test message {i}",
                timestamp=datetime.now(),
                user_id=f"user{i}",
                message_id=f"msg{i}"
            )
            for i in range(15)
        ]
        
        # Configure mocks for GPT-4
        def create_gpt4_response(*args, **kwargs):
            # Get the formatted messages from the API call
            if 'messages' in kwargs:
                messages = kwargs['messages']
                if isinstance(messages, list):
                    # Count the number of messages in the formatted string
                    content = messages[0]['content']
                    message_count = content.count('\n') + 1
                else:
                    message_count = 1
            else:
                message_count = 1  # For simple tests
            
            response = MagicMock()
            response.choices = [MagicMock()]
            # Create as many conversations as messages
            conversations = [{"id": f"conv{i}", "topic": f"test{i}", "confidence": 0.9} for i in range(message_count)]
            response.choices[0].message.content = json.dumps({"conversations": conversations})
            return response
            
        # Configure mocks for Claude
        def create_claude_response(*args, **kwargs):
            # Get the formatted messages from the API call
            if 'messages' in kwargs:
                messages = kwargs['messages']
                if isinstance(messages, list):
                    # Count the number of messages in the formatted string
                    content = messages[0]['content']
                    message_count = content.count('\n') + 1
                else:
                    message_count = 1
            else:
                message_count = 1  # For simple tests
            
            response = MagicMock()
            conversations = [{"id": f"conv{i}", "topic": f"test{i}", "confidence": 0.9} for i in range(message_count)]
            response.content = json.dumps({"conversations": conversations})
            return response
            
        # Initialize detectors with mocks
        with patch('openai.OpenAI') as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = create_gpt4_response
            self.gpt4_detector = GPT4ConversationDetector(
                api_key="test_key",
                batch_size=15,
                cache_dir=os.path.join(self.temp_dir, "gpt4_cache")
            )
            
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = create_claude_response
            self.claude_detector = Claude35ConversationDetector(
                api_key="test_key",
                batch_size=15,
                cache_dir=os.path.join(self.temp_dir, "claude35_cache")
            )
        
    def tearDown(self):
        """Clean up test environment after each test."""
        shutil.rmtree(self.temp_dir)
        
    def test_gpt4_detector_initialization(self):
        """Test GPT-4 detector initialization."""
        self.assertIsNotNone(self.gpt4_detector)
        self.assertIsNotNone(self.gpt4_detector.client)
        self.assertIsNotNone(self.gpt4_detector.cache_manager)
        
    def test_claude_detector_initialization(self):
        """Test Claude detector initialization."""
        self.assertIsNotNone(self.claude_detector)
        self.assertIsNotNone(self.claude_detector.client)
        self.assertIsNotNone(self.claude_detector.cache_manager)
        
    def test_gpt4_detector_caching(self):
        """Test GPT-4 detector caching functionality."""
        # Primera llamada
        results1 = self.gpt4_detector.detect(self.messages[:1])
        # Segunda llamada (debería usar caché)
        results2 = self.gpt4_detector.detect(self.messages[:1])
        
        self.assertEqual(len(results1), 1)
        self.assertEqual(len(results2), 1)
        self.assertEqual(
            results1[0].metadata["conversation_id"],
            results2[0].metadata["conversation_id"]
        )
        
    def test_claude_detector_caching(self):
        """Test Claude detector caching functionality."""
        # Primera llamada
        results1 = self.claude_detector.detect(self.messages[:1])
        # Segunda llamada (debería usar caché)
        results2 = self.claude_detector.detect(self.messages[:1])
        
        self.assertEqual(len(results1), 1)
        self.assertEqual(len(results2), 1)
        self.assertEqual(
            results1[0].metadata["conversation_id"],
            results2[0].metadata["conversation_id"]
        )
        
    def test_gpt4_detector_batch_processing(self):
        """Test GPT-4 detector batch processing."""
        results = self.gpt4_detector.detect(self.messages)
        self.assertEqual(len(results), len(self.messages))
        
    def test_claude_detector_batch_processing(self):
        """Test Claude detector batch processing."""
        results = self.claude_detector.detect(self.messages)
        self.assertEqual(len(results), len(self.messages))
        
    def test_gpt4_detector_empty_messages(self):
        """Test GPT-4 detector with empty messages."""
        results = self.gpt4_detector.detect([])
        self.assertEqual(len(results), 0)
        
    def test_claude_detector_empty_messages(self):
        """Test Claude detector with empty messages."""
        results = self.claude_detector.detect([])
        self.assertEqual(len(results), 0)
        
    def test_gpt4_detector_error_handling(self):
        """Test GPT-4 detector error handling."""
        with patch('openai.OpenAI') as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
            detector = GPT4ConversationDetector(
                api_key="test_key",
                cache_dir=os.path.join(self.temp_dir, "gpt4_cache_error")
            )
            results = detector.detect(self.messages[:1])
            self.assertEqual(len(results), 0)
            
    def test_claude_detector_error_handling(self):
        """Test Claude detector error handling."""
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")
            detector = Claude35ConversationDetector(
                api_key="test_key",
                cache_dir=os.path.join(self.temp_dir, "claude35_cache_error")
            )
            results = detector.detect(self.messages[:1])
            self.assertEqual(len(results), 0)
            
    def test_gpt4_detector_invalid_api_key(self):
        """Test GPT-4 detector with invalid API key."""
        with patch('openai.OpenAI') as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = Exception("Invalid API key")
            detector = GPT4ConversationDetector(
                api_key="invalid_key",
                cache_dir=os.path.join(self.temp_dir, "gpt4_cache_invalid")
            )
            results = detector.detect(self.messages[:1])
            self.assertEqual(len(results), 0)
            
    def test_claude_detector_invalid_api_key(self):
        """Test Claude detector with invalid API key."""
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = Exception("Invalid API key")
            detector = Claude35ConversationDetector(
                api_key="invalid_key",
                cache_dir=os.path.join(self.temp_dir, "claude35_cache_invalid")
            )
            results = detector.detect(self.messages[:1])
            self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main() 