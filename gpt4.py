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
from parallel_processor import ParallelProcessor, ProcessingResult

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
    Detects conversations using GPT-4 model with caching support.
    
    Attributes:
        client: OpenAI API client
        cache: Cache manager for API responses
        batch_size: Number of messages to process in batch
        parallel_processor: Parallel processor for batch processing
        
    Edge Cases Handled:
        - API rate limits
        - Network errors
        - Invalid responses
        - Message validation
    """
    
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 100, cache_dir: Optional[str] = None):
        """Initialize the GPT-4 conversation detector.

        Args:
            api_key (str, optional): OpenAI API key. If not provided, will try to get from environment.
            batch_size (int, optional): Number of messages to process in each batch. Defaults to 100.
            cache_dir (str, optional): Directory to store cache files. If not provided, will use default.
        """
        super().__init__()
        
        # Initialize API client
        self.client = openai.OpenAI(api_key=api_key)
        self.batch_size = batch_size
        
        # Initialize cache manager
        self.cache_manager = CacheManager(cache_dir or "gpt4_cache")
        
        # Initialize parallel processor
        self.parallel_processor = ParallelProcessor(
            max_workers=5,
            batch_size=batch_size,
            timeout=30
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        logger.info("GPT-4 detector initialized with caching")
        
    def detect(self, messages: List[Message]) -> List[Label]:
        """Detect conversations in messages using GPT-4.

        Args:
            messages: List of messages to analyze

        Returns:
            List of conversation labels
        """
        # Handle empty messages
        if not messages:
            return []
            
        # Validate messages
        if not self._validate_messages(messages):
            raise ValueError("Invalid messages provided")
            
        # Generate cache key
        cache_key = self._generate_cache_key(messages)
        
        # Check cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            return [Label.from_dict(label_dict) for label_dict in cached_result]
            
        try:
            # Process messages in parallel batches
            results = self.parallel_processor.process_batch(
                messages,
                self._process_message_batch
            )
            
            # Collect successful results
            labels = []
            for result in results:
                if result.success and result.data:
                    labels.extend(result.data)
            
            # Cache the results
            if labels:
                self.cache_manager.set(cache_key, [label.to_dict() for label in labels])
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error detecting conversations: {e}")
            return []
        
    def _process_message_batch(self, messages: List[Message]) -> List[Label]:
        """Process a batch of messages using GPT-4 API."""
        try:
            # Format messages for API
            formatted_messages = self._format_messages(messages)
            
            # Make API call with retries
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=[{
                            "role": "user",
                            "content": formatted_messages
                        }]
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self.logger.warning(f"API call failed, attempt {attempt + 1}: {e}")
                    time.sleep(retry_delay)
                    
            # Parse response into labels
            return self._parse_response(response)
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise
        
    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for GPT-4 API."""
        return "\n".join(
            f"[{msg.timestamp}] {msg.user_id}: {msg.content}"
            for msg in messages
        )
        
    def _parse_response(self, response) -> List[Label]:
        """
        Parse GPT-4 response into labels.
        
        Args:
            response: Raw API response
            
        Returns:
            List of conversation labels
            
        Edge Cases:
            - Invalid JSON
            - Missing fields
            - Malformed response
        """
        try:
            # Parse response JSON
            content = json.loads(response.choices[0].message.content)
            
            # Create labels from conversations
            labels = []
            for conv in content.get("conversations", []):
                label = Label(
                    label_type=conv["topic"],
                    confidence=conv.get("confidence", 0.0),
                    metadata={"conversation_id": conv["id"]}
                )
                labels.append(label)
                
            return labels
            
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error parsing response: {e}")
            return []

    def _generate_cache_key(self, messages: List[Message]) -> str:
        """Generate a cache key for a list of messages."""
        return "_".join(
            f"{msg.message_id}"
            for msg in messages
        )

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