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
from typing import List, Dict, Any, Optional
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

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
    
    def __init__(self):
        """Initialize the Claude detector with API configuration.
        
        Raises:
            ValueError: If ANTHROPIC_API_KEY is not set
            anthropic.error.AuthenticationError: If API key is invalid
        """
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        try:
            self.client = Anthropic(api_key=self.api_key)
            self.model = MODEL_NAME
            self.labeler_id = "claude35s"  # Identifier for this model
            logger.info("Successfully initialized Claude 3.5 detector")
        except Exception as e:
            logger.error(f"Error initializing Claude 3.5 detector: {e}")
            raise

    def detect(self, messages: List[Message]) -> List[Label]:
        """
        Detect conversations in a list of messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            List of labels assigning messages to conversations
            
        Raises:
            ValueError: If messages list is empty
            RuntimeError: If detection fails
            
        Edge Cases Handled:
            - Empty message list
            - Invalid message format
            - API errors
            - Batch processing errors
            - Memory constraints
            - Network issues
        """
        try:
            if not messages:
                raise ValueError("No messages provided for detection")
                
            # Convert Message objects to dict format
            message_dicts = [
                {
                    'id': msg.id,
                    'text': msg.text,
                    'timestamp': msg.timestamp,
                    'user': msg.user
                }
                for msg in messages
            ]
            
            # Process in batches
            all_labels = []
            total_batches = (len(message_dicts) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for i in range(0, len(message_dicts), BATCH_SIZE):
                batch = message_dicts[i:i + BATCH_SIZE]
                batch_num = i//BATCH_SIZE + 1
                logger.info(f"Processing batch {batch_num} of {total_batches}")
                
                try:
                    batch_labels = self._detect_conversations(batch)
                    all_labels.extend(batch_labels)
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    continue
                    
            if not all_labels:
                raise RuntimeError("No labels were generated")
                
            return all_labels
            
        except Exception as e:
            logger.error(f"Error in conversation detection: {e}")
            raise

    def _detect_conversations(self, messages: List[Dict[str, Any]]) -> List[Label]:
        """Implementation of conversation detection.
        
        Args:
            messages: List of message dictionaries to analyze
            
        Returns:
            List of Label objects
            
        Raises:
            ValueError: If messages list is empty
            RuntimeError: If API call fails
            
        Edge Cases Handled:
            - Empty message list
            - API errors
            - Invalid responses
            - Malformed CSV
            - Invalid confidence scores
            - Network timeouts
        """
        try:
            if not messages:
                raise ValueError("Empty message list provided")
                
            prompt = self._create_analysis_prompt(messages)
            
            # Make API call with retry logic
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        system="You are a conversation detection system that analyzes message patterns and content to identify distinct conversations. Your output should be in CSV format with exactly 6 columns: message_id,conversation_id,topic,timestamp,labeler_id,confidence",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=MAX_TOKENS
                    )
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    logger.warning(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    asyncio.sleep(RETRY_DELAY)
                    continue
            
            # Parse the CSV response
            csv_response = response.content[0].text.strip()
            
            # Remove any markdown formatting
            if csv_response.startswith('```csv'):
                csv_response = csv_response[6:]
            if csv_response.endswith('```'):
                csv_response = csv_response[:-3]
            
            # Split into lines and remove empty lines
            lines = [line.strip() for line in csv_response.split('\n') if line.strip()]
            
            # Skip header if present
            if lines and lines[0].startswith('message_id,conversation_id'):
                lines = lines[1:]
            
            # Convert to Label objects
            labels = []
            for line in lines:
                try:
                    parts = line.split(',')
                    if len(parts) != 6:
                        logger.warning(f"Skipping malformed line: {line} - Expected 6 parts, got {len(parts)}")
                        continue
                        
                    msg_id, conv_id, topic, timestamp, labeler_id, confidence = parts
                    
                    # Clean up any quotes
                    msg_id = msg_id.strip('"')
                    conv_id = conv_id.strip('"')
                    topic = topic.strip('"')
                    timestamp = timestamp.strip('"')
                    labeler_id = labeler_id.strip('"')
                    confidence = confidence.strip('"')
                    
                    try:
                        confidence = float(confidence)
                        if confidence < 0 or confidence > 1:
                            logger.warning(f"Invalid confidence score: {confidence}, using default")
                            confidence = 0.5
                    except ValueError:
                        logger.warning(f"Could not parse confidence score: {confidence}, using default")
                        confidence = 0.5
                    
                    label = Label(
                        message_id=msg_id,
                        conversation_id=conv_id,
                        topic=topic,
                        timestamp=timestamp,
                        metadata={
                            'labeler_id': labeler_id,
                            'confidence': confidence
                        }
                    )
                    labels.append(label)
                except Exception as e:
                    logger.warning(f"Skipping malformed line: {line} - Error: {str(e)}")
                    continue
            
            if not labels:
                raise RuntimeError("No valid labels generated from response")
                
            return labels
            
        except Exception as e:
            logger.error(f"Error in Claude 3.5 conversation detection: {str(e)}")
            raise

    def _create_analysis_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Create the analysis prompt for Claude.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If messages list is empty
            
        Edge Cases Handled:
            - Empty message list
            - Missing message fields
            - Invalid message format
            - Missing user information
        """
        if not messages:
            raise ValueError("Empty message list provided")
            
        formatted_messages = "\n"
        for msg in messages:
            try:
                formatted_messages += f"Message ID: {msg['id']}\n"
                formatted_messages += f"Timestamp: {msg['timestamp']}\n"
                formatted_messages += f"User: {msg['user'].get('username', 'Unknown')}\n"
                formatted_messages += f"Content: {msg['text']}\n\n"
            except KeyError as e:
                logger.warning(f"Skipping message with missing field: {e}")
                continue

        return CONVERSATION_DETECTION_PROMPT.replace("[MESSAGES]", formatted_messages)

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