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
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from ..utils.cache_manager import CacheManager

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
        cache_manager: Cache manager instance for storing results
    """
    
    def __init__(self, batch_size: int = 10, cache_dir: str = "cache"):
        """
        Initialize the conversation detector.
        
        Args:
            batch_size: Number of messages to process in each batch
            cache_dir: Directory for cache storage
        """
        self.batch_size = batch_size
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        
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

    def _get_cache_key(self, messages: List[Message]) -> str:
        """
        Generate a cache key for a list of messages.
        
        Args:
            messages: List of messages to generate key for
            
        Returns:
            Cache key string
        """
        # Create a unique key based on message IDs and content
        message_data = [(msg.message_id, msg.content) for msg in messages]
        return f"conv_detect_{hash(str(message_data))}"

    def _get_cached_result(self, messages: List[Message]) -> Optional[List[Label]]:
        """
        Get cached detection results for messages.
        
        Args:
            messages: List of messages to check cache for
            
        Returns:
            Cached labels if found, None otherwise
        """
        cache_key = self._get_cache_key(messages)
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data:
            try:
                return [Label.from_dict(label_data) for label_data in cached_data]
            except Exception:
                return None
        return None

    def _cache_result(self, messages: List[Message], labels: List[Label]) -> None:
        """
        Cache detection results for messages.
        
        Args:
            messages: List of messages
            labels: List of labels to cache
        """
        cache_key = self._get_cache_key(messages)
        label_data = [label.to_dict() for label in labels]
        self.cache_manager.set(cache_key, label_data)

    def clear_cache(self) -> None:
        """
        Clear the detector's cache.
        """
        self.cache_manager.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return self.cache_manager.get_stats() 