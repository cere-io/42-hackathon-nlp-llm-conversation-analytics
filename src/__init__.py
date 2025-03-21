"""
Conversation Analytics System

A comprehensive system for detecting and analyzing conversations in chat messages
using natural language processing and machine learning techniques.
"""

from .detectors.conversation_detector import ConversationDetector
from .processors.text_vectorizer import TextVectorizer
from .metrics.conversation_metrics import evaluate_conversations

__version__ = "0.1.0"
__all__ = ["ConversationDetector", "TextVectorizer", "evaluate_conversations"] 