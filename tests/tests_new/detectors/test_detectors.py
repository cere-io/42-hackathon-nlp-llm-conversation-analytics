"""
Tests for all detector classes including conversation detector and spam detector.
"""

import numpy as np
import pytest
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from src.detectors.conversation_detector import ConversationDetector
from src.detectors.spam_detector import SpamDetector

# Fixtures
@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {
            "text": "Hello",
            "timestamp": "2024-01-01 10:00:00",
            "user": "user1"
        },
        {
            "text": "Hi there",
            "timestamp": "2024-01-01 10:00:05",
            "user": "user2"
        },
        {
            "text": "How are you?",
            "timestamp": "2024-01-01 10:00:10",
            "user": "user1"
        },
        {
            "text": "Good morning",
            "timestamp": "2024-01-01 10:05:00",
            "user": "user3"
        },
        {
            "text": "Morning!",
            "timestamp": "2024-01-01 10:05:05",
            "user": "user4"
        },
        {
            "text": "Have a good day",
            "timestamp": "2024-01-01 10:10:00",
            "user": "user1"
        }
    ]

@pytest.fixture
def sample_vectors():
    """Create sample vectors for testing."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0]
    ])

@pytest.fixture
def sample_texts():
    """Create sample texts for spam detection testing."""
    return [
        "Hello, how are you?",
        "Buy now! Limited time offer!",
        "Congratulations! You've won!",
        "Meeting at 3 PM tomorrow",
        "URGENT: Your account needs verification",
        "Let's discuss the project",
        "FREE VIAGRA NOW!!!",
        "Important: Project deadline extended"
    ]

@pytest.fixture
def sample_labels():
    """Create sample labels for spam detection testing."""
    return [0, 1, 1, 0, 1, 0, 1, 0]  # 0: ham, 1: spam

# Conversation Detector Tests
def test_conversation_detector_initialization():
    """Test conversation detector initialization."""
    detector = ConversationDetector()
    assert detector.time_threshold == 300  # 5 minutes in seconds
    assert detector.similarity_threshold == 0.5
    assert detector.min_conversation_size == 2
    assert detector.max_conversation_size == 50

def test_conversation_detector_detect(sample_messages, sample_vectors):
    """Test conversation detection."""
    detector = ConversationDetector()
    conversations = detector.detect(sample_messages, sample_vectors)
    
    # Check that conversations were detected
    assert len(conversations) > 0
    
    # Check conversation structure
    for conv in conversations:
        assert 'messages' in conv
        assert 'start_time' in conv
        assert 'end_time' in conv
        assert 'participants' in conv
        assert len(conv['messages']) >= detector.min_conversation_size
        assert len(conv['messages']) <= detector.max_conversation_size

def test_conversation_detector_time_based(sample_messages, sample_vectors):
    """Test time-based conversation detection."""
    detector = ConversationDetector(time_threshold=60)  # 1 minute threshold
    
    # Modify timestamps to create time gaps
    messages = sample_messages.copy()
    messages[3]['timestamp'] = "2024-01-01 10:02:00"  # 2 minutes after previous message
    
    conversations = detector.detect(messages, sample_vectors)
    
    # Should detect at least 2 conversations due to time gap
    assert len(conversations) >= 2
    
    # Check conversation times
    for conv in conversations:
        start_time = datetime.strptime(conv['start_time'], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(conv['end_time'], "%Y-%m-%d %H:%M:%S")
        duration = (end_time - start_time).total_seconds()
        assert duration <= detector.time_threshold

# Spam Detector Tests
def test_spam_detector_initialization():
    """Test spam detector initialization."""
    detector = SpamDetector()
    assert isinstance(detector.classifier, RandomForestClassifier)
    assert detector.threshold == 0.5

def test_spam_detector_fit_predict(sample_texts, sample_labels):
    """Test spam detection."""
    detector = SpamDetector()
    detector.fit(sample_texts, sample_labels)
    
    # Test prediction
    predictions = detector.predict(sample_texts)
    assert len(predictions) == len(sample_texts)
    assert all(pred in [0, 1] for pred in predictions)
    
    # Test prediction probabilities
    probabilities = detector.predict_proba(sample_texts)
    assert probabilities.shape == (len(sample_texts), 2)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)

def test_spam_detector_threshold(sample_texts, sample_labels):
    """Test spam detection threshold."""
    detector = SpamDetector(threshold=0.7)
    detector.fit(sample_texts, sample_labels)
    
    # Get probabilities
    probabilities = detector.predict_proba(sample_texts)
    
    # Check threshold-based predictions
    predictions = detector.predict(sample_texts)
    for i, prob in enumerate(probabilities[:, 1]):
        if prob >= detector.threshold:
            assert predictions[i] == 1
        else:
            assert predictions[i] == 0

def test_spam_detector_features(sample_texts, sample_labels):
    """Test feature extraction."""
    detector = SpamDetector()
    detector.fit(sample_texts, sample_labels)
    
    # Check feature importance
    feature_importance = detector.get_feature_importance()
    assert len(feature_importance) > 0
    assert all(imp >= 0 for imp in feature_importance.values())
    
    # Check top spam words
    top_spam_words = detector.get_top_spam_words(n=5)
    assert len(top_spam_words) <= 5
    assert all(word in feature_importance for word in top_spam_words)

# Error Handling Tests
def test_detectors_empty_data():
    """Test handling of empty data."""
    detectors = [
        ConversationDetector(),
        SpamDetector()
    ]
    
    for detector in detectors:
        with pytest.raises(ValueError):
            if isinstance(detector, ConversationDetector):
                detector.detect([], np.array([]))
            else:
                detector.fit([], [])

def test_detectors_mismatched_data():
    """Test handling of mismatched data lengths."""
    conversation_detector = ConversationDetector()
    messages = [{"text": "Hello", "timestamp": "2024-01-01 10:00:00", "user": "user1"}]
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    with pytest.raises(ValueError):
        conversation_detector.detect(messages, vectors)
    
    spam_detector = SpamDetector()
    with pytest.raises(ValueError):
        spam_detector.fit(["text1", "text2"], [0])

def test_detectors_invalid_timestamps():
    """Test handling of invalid timestamps."""
    detector = ConversationDetector()
    messages = [
        {
            "text": "Hello",
            "timestamp": "invalid",
            "user": "user1"
        }
    ]
    vectors = np.array([[1.0, 0.0, 0.0]])
    
    with pytest.raises(ValueError):
        detector.detect(messages, vectors)

# Persistence Tests
def test_detectors_persistence(tmp_path, sample_texts, sample_labels):
    """Test detector persistence."""
    # Test spam detector persistence
    spam_detector = SpamDetector()
    spam_detector.fit(sample_texts, sample_labels)
    
    save_path = tmp_path / "spam_detector.pkl"
    spam_detector.save(save_path)
    
    loaded_spam_detector = SpamDetector.load(save_path)
    predictions1 = spam_detector.predict(sample_texts)
    predictions2 = loaded_spam_detector.predict(sample_texts)
    assert np.array_equal(predictions1, predictions2) 