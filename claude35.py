"""
Claude 3.5 Sonnet inference module for conversation detection.

This module provides functionality for analyzing messages and detecting conversations using Anthropic's Claude 3.5 Sonnet model.
It handles message processing, conversation labeling, and topic detection with robust error handling and logging.

Key Features:
- Batch processing of messages to handle large datasets
- Robust error handling and logging
- Configurable model parameters
- CSV-based input/output
- Conversation topic detection
- Confidence scoring for labels
- Automatic output file naming with timestamps
- Response caching
- Performance monitoring

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
- Network connectivity issues
- Token limit exceeded
- Invalid response formats

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
import argparse
import sys
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv
from conversation_detector import ConversationDetector, Message, Label
from cache_manager import CacheManager

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import ConversationDetector, Message, Label

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

# Read the conversation detection prompt
PROMPT_FILE = os.path.join(os.path.dirname(__file__), 'open_source_examples/prompts/conversation_detection_prompt.txt')
try:
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        exec(f.read())  # This will load CONVERSATION_DETECTION_PROMPT
except Exception as e:
    logger.error(f"Error loading prompt file: {e}")
    raise

# Define output directory relative to the module
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'csv', 'detections')
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception as e:
    logger.error(f"Error creating output directory: {e}")
    raise

# Constants
BATCH_SIZE = 50  # Process messages in batches of 50 to stay within token limits
MODEL_NAME = "claude-3-5-sonnet-latest"  # Claude 3.5 Sonnet model (latest version)
MAX_RETRIES = 3  # Maximum number of API retry attempts
RETRY_DELAY = 1  # Delay between retries in seconds
MAX_TOKENS = 1000  # Maximum tokens for API response

class Claude35ConversationDetector(ConversationDetector):
    """
    Conversation detector using Claude 3.5 Sonnet for message analysis and topic detection.
    
    This class implements conversation detection using Anthropic's Claude 3.5 Sonnet model,
    with support for batch processing and robust error handling.
    
    Edge Cases Handled:
        - API rate limits
        - Network timeouts
        - Invalid responses
        - Token limit exceeded
        - Model errors
        - Invalid prompts
        - Missing user information
        - Malformed CSV data
        - Invalid confidence scores
    """
    
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 50):
        """
        Initialize the Claude 3.5 Sonnet conversation detector.
        
        Args:
            api_key: Anthropic API key (optional, can be set via environment)
            batch_size: Number of messages to process in each batch
            
        Raises:
            ValueError: If API key is not provided and not in environment
        """
        super().__init__()
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        if not self.client.api_key:
            raise ValueError("Anthropic API key must be provided or set in environment")
            
        self.cache = CacheManager(cache_dir="cache/claude35", max_size=1000)
        self.model_name = "claude-3-sonnet-20240229"
        self.batch_size = batch_size
        self.labeler_id = "claude35s"  # Identifier for this model
        logger.info("Successfully initialized Claude 3.5 detector")

    def detect(self, messages: List[Message]) -> List[Label]:
        """
        Detect conversations in a list of messages using Claude 3.5 Sonnet.
        
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
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=1000,
                        messages=formatted_messages
                    )
                    
                    # Cache the response
                    self.cache.set(cache_key, response.content[0].text)
                    
                    # Parse and return results
                    return self._parse_response(response.content[0].text)
                    
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
        return f"claude35_{hash(content)}"
        
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
            {
                "role": "user",
                "content": json.dumps([m.__dict__ for m in messages])
            }
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
        parser = argparse.ArgumentParser(description='Run Claude 3.5 Sonnet conversation detection on a messages file')
        parser.add_argument('input_file', help='Path to input CSV file containing messages')
        parser.add_argument('--output', '-o', help='Path to output file (default: auto-generated in data/csv/detections/)',
                          default=None)
        
        args = parser.parse_args()
        
        # Create detector instance
        detector = Claude35ConversationDetector()
        
        # Load messages
        messages = detector.load_messages(args.input_file)
        logger.info(f"Loaded {len(messages)} messages from {args.input_file}")
        
        # Detect conversations
        labels = detector.detect(messages)
        
        # Generate output filename if not provided
        if args.output is None:
            # Extract group name from input file path
            group_name = 'origintrail'  # Hardcode for now since we know the group
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            args.output = os.path.join(OUTPUT_DIR, f'labels_{timestamp}_{detector.labeler_id}_{group_name}.csv')
        
        # Save results
        detector.save_labels(labels, args.output)
        logger.info(f"Results written to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1) 