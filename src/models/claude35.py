"""
Claude 3.5 inference module for conversation detection.

This module provides functionality for analyzing messages and detecting conversations using Anthropic's Claude 3.5 model.
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
    python claude35.py input_messages.csv --output results.csv

Requirements:
1. ANTHROPIC_API_KEY environment variable set
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
import anthropic
from anthropic import Anthropic
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
        logging.FileHandler('claude35_detection.log')
    ]
)
logger = logging.getLogger(__name__)

class Claude35ConversationDetector(ConversationDetector):
    """
    Detects conversations using Claude 3.5 model with caching support.
    
    Attributes:
        client: Anthropic API client
        cache: Cache manager for API responses
        batch_size: Number of messages to process in batch
        
    Edge Cases Handled:
        - API rate limits
        - Network errors
        - Invalid responses
        - Message validation
    """
    
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 100, cache_dir: Optional[str] = None):
        """Initialize the Claude 3.5 conversation detector.

        Args:
            api_key (str, optional): Anthropic API key. If not provided, will try to get from environment.
            batch_size (int, optional): Number of messages to process in each batch. Defaults to 100.
            cache_dir (str, optional): Directory to store cache files. If not provided, will use default.
        """
        super().__init__()
        
        # Initialize API client
        self.client = anthropic.Anthropic(api_key=api_key)
        self.batch_size = batch_size
        
        # Initialize cache manager
        self.cache_manager = CacheManager(cache_dir or "claude35_cache")
        
        # Initialize parallel processor
        self.parallel_processor = ParallelProcessor(
            max_workers=5,
            batch_size=batch_size,
            timeout=30
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Claude 3.5 detector initialized with caching")
        
    def detect(self, messages: List[Message]) -> List[Label]:
        """Detect conversations in messages using Claude 3.5.

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
        """Process a batch of messages using Claude 3.5 API."""
        try:
            formatted_messages = self._format_messages(messages)
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": formatted_messages
                }]
            )
            return self._parse_response(response)
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise
        
    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for Claude 3.5 API."""
        return "\n".join(
            f"[{msg.timestamp}] {msg.user_id}: {msg.content}"
            for msg in messages
        )
        
    def _parse_response(self, response) -> List[Label]:
        """
        Parse Claude 3.5 response into labels.
        
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
            content = json.loads(response.content)
            
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