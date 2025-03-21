"""
Test Detector Module

This module provides a test implementation of the ConversationDetector
for testing purposes.
"""

from typing import List
from datetime import datetime
from .conversation_detector import ConversationDetector, Message, Label

class TestDetector(ConversationDetector):
    """
    Test implementation of ConversationDetector that uses caching.
    """
    
    def detect(self, messages: List[Message]) -> List[Label]:
        """
        Detect and label conversations in a list of messages.
        This is a test implementation that uses caching.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            List of labels for the conversations
        """
        if not self._validate_messages(messages):
            return []
            
        # Check cache first
        cached_labels = self._get_cached_result(messages)
        if cached_labels is not None:
            return cached_labels
            
        # If not in cache, process messages
        # This is a simple test implementation
        labels = []
        for msg in messages:
            # Simple test logic: if message contains "test", label as "test"
            label_type = "test" if "test" in msg.content.lower() else "normal"
            label = Label(
                label_type=label_type,
                confidence=0.8,
                metadata={"processed_at": datetime.now().isoformat()}
            )
            labels.append(label)
            
        # Cache the results
        self._cache_result(messages, labels)
        
        return labels 