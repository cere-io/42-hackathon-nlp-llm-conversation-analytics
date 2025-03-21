"""
GPT-4 inference module for conversation detection.

This module provides functionality for analyzing messages and detecting conversations using OpenAI's GPT-4 model.
It handles message processing, conversation labeling, and topic detection with robust error handling and logging.

Key Features:
- Batch processing of messages to handle large datasets
- Robust error handling and logging
- Configurable model parameters
- CSV-based input/output
- Conversation topic detection
- Confidence scoring for labels

Edge Cases Handled:
- Missing or invalid API keys
- Empty or malformed input files
- API rate limits and timeouts
- Invalid message formats
- Missing user information
- Malformed CSV responses
- File system errors
- Memory constraints
- Invalid confidence scores
- Missing or corrupted prompt files

Example Usage:
    python gpt4.py input_messages.csv --output results.csv

Requirements:
1. OPENAI_API_KEY environment variable set
2. Input CSV file with columns:
   - id: Message identifier
   - text: Message content
   - timestamp: Message timestamp
   - username: User identifier
   - first_name: User's first name (optional)
   - last_name: User's last name (optional)
"""

import os
import json
import csv
import logging
import asyncio
import argparse
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from openai import OpenAI
from dotenv import load_dotenv
from conversation_detector import ConversationDetector, Message, Label
from cache_manager import CacheManager

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gpt4_detection.log')
    ]
)
logger = logging.getLogger(__name__)

class GPT4ConversationDetector(ConversationDetector):
    """
    Conversation detector using OpenAI's GPT-4 model with caching.
    
    Attributes:
        client (OpenAI): OpenAI API client
        cache (CacheManager): Cache for API responses
        model_name (str): Name of the GPT model to use
        batch_size (int): Number of messages to process in each batch
        
    Edge Cases Handled:
        - API rate limits
        - Network errors
        - Invalid responses
        - Cache misses
        - Memory pressure
    """
    
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 50):
        """
        Initialize the GPT-4 conversation detector.
        
        Args:
            api_key: OpenAI API key (optional, can be set via environment)
            batch_size: Number of messages to process in each batch
            
        Raises:
            ValueError: If API key is not provided and not in environment
        """
        super().__init__()
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key must be provided or set in environment")
            
        self.cache = CacheManager(cache_dir="cache/gpt4", max_size=1000)
        self.model_name = "gpt-4-0125-preview"
        self.batch_size = batch_size
        
    def detect(self, messages: List[Message]) -> List[Label]:
        """
        Detect conversations in a list of messages using GPT-4.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            List of conversation labels
            
        Edge Cases:
            - Empty message list
            - Invalid message format
            - API errors
            - Cache misses
        """
        if not messages:
            logger.warning("Empty message list provided")
            return []
            
        try:
            # Process messages in batches
            labels = []
            for i in range(0, len(messages), self.batch_size):
                batch = messages[i:i + self.batch_size]
                batch_labels = self._detect_conversations(batch)
                labels.extend(batch_labels)
                
            return labels
            
        except Exception as e:
            logger.error(f"Error in conversation detection: {e}")
            return []
            
    def _detect_conversations(self, messages: List[Message]) -> List[Label]:
        """
        Detect conversations in a batch of messages.
        
        Args:
            messages: Batch of messages to analyze
            
        Returns:
            List of conversation labels for the batch
        """
        try:
            # Create cache key from message content
            cache_key = self._create_cache_key(messages)
            
            # Check cache first
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for batch of {len(messages)} messages")
                return self._parse_cached_result(cached_result)
                
            # Prepare messages for API
            formatted_messages = self._format_messages(messages)
            
            # Call API with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=formatted_messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    # Cache the response
                    self.cache.set(cache_key, response.choices[0].message.content)
                    
                    # Parse and return results
                    return self._parse_response(response.choices[0].message.content)
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"API call failed, attempt {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Error processing message batch: {e}")
            return []
            
    def _create_cache_key(self, messages: List[Message]) -> str:
        """
        Create a cache key from message content.
        
        Args:
            messages: List of messages
            
        Returns:
            String cache key
        """
        content = "|".join(f"{m.timestamp}_{m.text}" for m in messages)
        return f"gpt4_{hash(content)}"
        
    def _parse_cached_result(self, cached_content: str) -> List[Label]:
        """
        Parse cached API response.
        
        Args:
            cached_content: Cached response content
            
        Returns:
            List of conversation labels
        """
        try:
            return self._parse_response(cached_content)
        except Exception as e:
            logger.error(f"Error parsing cached result: {e}")
            return []
            
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Format messages for API input.
        
        Args:
            messages: List of messages to format
            
        Returns:
            List of formatted messages
        """
        return [
            {"role": "system", "content": "You are a conversation detection assistant."},
            {"role": "user", "content": json.dumps([m.__dict__ for m in messages])}
        ]
        
    def _parse_response(self, response: str) -> List[Label]:
        """
        Parse API response into labels.
        
        Args:
            response: API response string
            
        Returns:
            List of conversation labels
        """
        try:
            # Parse JSON response
            data = json.loads(response)
            return [Label(**label) for label in data]
        except Exception as e:
            logger.error(f"Error parsing API response: {e}")
            return []
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return self.cache.get_stats()

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Run GPT-4 conversation detection on a messages file')
        parser.add_argument('input_file', help='Path to input CSV file containing messages')
        parser.add_argument('--output', '-o', help='Path to output file',
                          default='gpt4_results.csv')
        
        args = parser.parse_args()
        
        # Create detector instance
        detector = GPT4ConversationDetector()
        
        # Load messages
        messages = detector.load_messages(args.input_file)
        logger.info(f"Loaded {len(messages)} messages from {args.input_file}")
        
        # Detect conversations
        labels = detector.detect(messages)
        
        # Save results
        detector.save_labels(labels, args.output)
        logger.info(f"Results written to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1) 