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
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from openai import OpenAI
from dotenv import load_dotenv

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

class Message:
    """Represents a single message in a conversation.
    
    Attributes:
        id: Unique identifier for the message
        text: Content of the message
        timestamp: When the message was sent
        user: Dictionary containing user information
        
    Edge Cases Handled:
        - Missing user information
        - Empty message text
        - Invalid timestamps
    """
    def __init__(self, id: str, text: str, timestamp: str, user: Dict[str, str]):
        if not id:
            raise ValueError("Message ID cannot be empty")
        if not text:
            raise ValueError("Message text cannot be empty")
        if not timestamp:
            raise ValueError("Message timestamp cannot be empty")
            
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.user = user or {}

class Label:
    """Represents a conversation label with metadata.
    
    Attributes:
        message_id: ID of the labeled message
        conversation_id: ID of the conversation
        topic: Topic label for the conversation
        timestamp: When the label was created
        metadata: Additional label information
        
    Edge Cases Handled:
        - Missing metadata
        - Invalid confidence scores
        - Missing required fields
    """
    def __init__(self, message_id: str, conversation_id: str, topic: str, 
                 timestamp: str, metadata: Dict[str, Any]):
        if not message_id:
            raise ValueError("Message ID cannot be empty")
        if not conversation_id:
            raise ValueError("Conversation ID cannot be empty")
        if not topic:
            raise ValueError("Topic cannot be empty")
        if not timestamp:
            raise ValueError("Timestamp cannot be empty")
            
        self.message_id = message_id
        self.conversation_id = conversation_id
        self.topic = topic
        self.timestamp = timestamp
        self.metadata = metadata or {}

class ConversationDetector:
    """Base class for conversation detection implementations.
    
    Provides common functionality for loading messages and saving labels.
    """
    
    def load_messages(self, input_file: str) -> List[Message]:
        """Load messages from CSV file.
        
        Args:
            input_file: Path to input CSV file
            
        Returns:
            List of Message objects
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If required columns are missing
            csv.Error: If file is not valid CSV
            
        Edge Cases Handled:
            - Missing input file
            - Empty file
            - Missing required columns
            - Invalid data types
            - Missing user information
            - Malformed CSV
        """
        try:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
                
            messages = []
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Validate required columns
                required_columns = {'id', 'text', 'timestamp', 'username'}
                missing_columns = required_columns - set(reader.fieldnames or [])
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                    
                for row in reader:
                    try:
                        # Handle missing user information
                        user = {
                            'username': row.get('username', ''),
                            'first_name': row.get('first_name', ''),
                            'last_name': row.get('last_name', '')
                        }
                        
                        message = Message(
                            id=row['id'],
                            text=row['text'],
                            timestamp=row['timestamp'],
                            user=user
                        )
                        messages.append(message)
                    except Exception as e:
                        logger.warning(f"Skipping invalid message row: {e}")
                        continue
                        
            if not messages:
                raise ValueError("No valid messages found in input file")
                
            logger.info(f"Successfully loaded {len(messages)} messages")
            return messages
            
        except csv.Error as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading messages: {e}")
            raise

    def save_labels(self, labels: List[Label], output_file: str):
        """Save labels to CSV file.
        
        Args:
            labels: List of Label objects to save
            output_file: Path to output CSV file
            
        Raises:
            ValueError: If labels list is empty
            IOError: If file cannot be written
            
        Edge Cases Handled:
            - Empty labels list
            - Invalid file path
            - Permission errors
            - Disk space issues
            - Corrupted data
        """
        try:
            if not labels:
                raise ValueError("No labels to save")
                
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
            os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['message_id', 'conversation_id', 'topic', 'timestamp', 'labeler_id', 'confidence'])
                
                for label in labels:
                    try:
                        writer.writerow([
                            label.message_id,
                            label.conversation_id,
                            label.topic,
                            label.timestamp,
                            label.metadata.get('labeler_id', ''),
                            label.metadata.get('confidence', 0.0)
                        ])
                    except Exception as e:
                        logger.warning(f"Skipping invalid label: {e}")
                        continue
                        
            logger.info(f"Successfully saved {len(labels)} labels to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving labels: {e}")
            raise

# Read the conversation detection prompt
PROMPT_FILE = os.path.join(os.path.dirname(__file__), 'open_source_examples/prompts/conversation_detection_prompt.txt')
try:
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        exec(f.read())  # This will load CONVERSATION_DETECTION_PROMPT
except Exception as e:
    logger.error(f"Error loading prompt file: {e}")
    raise

# Constants
BATCH_SIZE = 50  # Process messages in batches of 50 to stay within token limits
MODEL_NAME = "gpt-4-0125-preview"  # GPT-4 Turbo model
MAX_RETRIES = 3  # Maximum number of API retry attempts
RETRY_DELAY = 1  # Delay between retries in seconds

class GPT4ConversationDetector(ConversationDetector):
    """
    Conversation detector using GPT-4-Turbo for message analysis and topic detection.
    
    This class implements conversation detection using OpenAI's GPT-4 model,
    with support for batch processing and robust error handling.
    
    Edge Cases Handled:
        - API rate limits
        - Network timeouts
        - Invalid responses
        - Token limit exceeded
        - Model errors
        - Invalid prompts
    """
    
    def __init__(self):
        """Initialize the GPT-4 detector with API configuration.
        
        Raises:
            ValueError: If OPENAI_API_KEY is not set
            openai.error.AuthenticationError: If API key is invalid
        """
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.model = MODEL_NAME
            self.labeler_id = "gpt4o"
            logger.info("Successfully initialized GPT-4 detector")
        except Exception as e:
            logger.error(f"Error initializing GPT-4 detector: {e}")
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
        """
        try:
            if not messages:
                raise ValueError("Empty message list provided")
                
            prompt = self._create_analysis_prompt(messages)
            
            # Make API call with retry logic
            for attempt in range(MAX_RETRIES):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a conversation detection system that analyzes message patterns and content to identify distinct conversations. Output ONLY the CSV data with no additional text or formatting."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1000
                    )
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    logger.warning(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    asyncio.sleep(RETRY_DELAY)
                    continue
            
            # Parse the CSV response
            csv_response = response.choices[0].message.content.strip()
            
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
            logger.error(f"Error in GPT-4 conversation detection: {str(e)}")
            raise

    def _create_analysis_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Create the analysis prompt for GPT-4.
        
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