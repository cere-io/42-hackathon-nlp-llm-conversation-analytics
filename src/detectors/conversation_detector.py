"""
Conversation Detector Module

This module provides base classes for conversation detection and analysis.
It defines the core data structures and interfaces used by specific detector
implementations like GPT-4 and Claude 3.5.

Classes:
    Message: Represents a single message in a conversation
    Label: Represents a conversation label/classification
    ConversationDetector: Base class for conversation detectors
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import json

@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        content: The text content of the message
        timestamp: When the message was sent
        user_id: Identifier of the message sender
        message_id: Unique identifier for the message
    """
    content: str
    timestamp: datetime
    user_id: str
    message_id: str
    
    def to_dict(self) -> dict:
        """Convert message to dictionary format."""
        return {
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'message_id': self.message_id
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        """Create message from dictionary format."""
        return cls(
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data['user_id'],
            message_id=data['message_id']
        )

@dataclass
class Label:
    """
    Represents a conversation label/classification.
    
    Attributes:
        label_type: Type of conversation (e.g., 'spam', 'support', 'sales')
        confidence: Confidence score for the label
        metadata: Additional label information
    """
    label_type: str
    confidence: float
    metadata: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert label to dictionary format."""
        return {
            'label_type': self.label_type,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Label':
        """Create label from dictionary format."""
        return cls(
            label_type=data['label_type'],
            confidence=data['confidence'],
            metadata=data.get('metadata')
        )

class ConversationDetector:
    """
    Base class for conversation detectors.
    
    This class defines the interface that all conversation detectors must implement.
    Specific implementations (like GPT-4 or Claude 3.5) will extend this class.
    
    Attributes:
        batch_size: Number of messages to process in each batch
    """
    
    def __init__(self, batch_size: int = 10):
        """
        Initialize the conversation detector.
        
        Args:
            batch_size: Number of messages to process in each batch
        """
        self.batch_size = batch_size
        
    def detect(self, messages: List[Message]) -> List[Label]:
        """
        Detect and label conversations in a list of messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            List of labels for the conversations
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement detect()")
        
    def _validate_messages(self, messages: List[Message]) -> bool:
        """
        Validate a list of messages.
        
        Args:
            messages: List of messages to validate
            
        Returns:
            True if messages are valid, False otherwise
        """
        if not messages:
            return False
            
        try:
            for msg in messages:
                if not isinstance(msg, Message):
                    return False
                if not msg.content or not msg.user_id or not msg.message_id:
                    return False
        except Exception:
            return False
            
        return True 