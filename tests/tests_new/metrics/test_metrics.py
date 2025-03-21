"""
Tests for all metrics classes including conversation metrics and clustering metrics.
"""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.metrics.conversation_metrics import evaluate_conversations
from src.metrics.clustering_metrics import calculate_clustering_metrics

# Fixtures
@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
    return [
        {
            "messages": [
                {"text": "Hello", "timestamp": "2024-01-01 10:00:00", "user": "user1"},
                {"text": "Hi", "timestamp": "2024-01-01 10:00:05", "user": "user2"},
                {"text": "How are you?", "timestamp": "2024-01-01 10:00:10", "user": "user1"}
            ],
            "start_time": "2024-01-01 10:00:00",
            "end_time": "2024-01-01 10:00:10",
            "participants": {"user1", "user2"}
        },
        {
            "messages": [
                {"text": "Good morning", "timestamp": "2024-01-01 10:05:00", "user": "user3"},
                {"text": "Morning!", "timestamp": "2024-01-01 10:05:05", "user": "user4"}
            ],
            "start_time": "2024-01-01 10:05:00",
            "end_time": "2024-01-01 10:05:05",
            "participants": {"user3", "user4"}
        }
    ]

@pytest.fixture
def sample_data():
    """Create sample data for clustering metrics testing."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0]
    ])

@pytest.fixture
def sample_labels():
    """Create sample labels for clustering metrics testing."""
    return np.array([0, 0, 0, 1, 1, 2])

# Conversation Metrics Tests
def test_evaluate_conversations_basic(sample_conversations):
    """Test basic conversation evaluation."""
    metrics = evaluate_conversations(sample_conversations)
    
    # Check required metrics
    assert 'num_conversations' in metrics
    assert 'avg_conversation_length' in metrics
    assert 'avg_participants' in metrics
    assert 'total_messages' in metrics
    
    # Check metric values
    assert metrics['num_conversations'] == 2
    assert metrics['total_messages'] == 5
    assert 2 <= metrics['avg_participants'] <= 3
    assert metrics['avg_conversation_length'] > 0

def test_evaluate_conversations_time_metrics(sample_conversations):
    """Test time-based metrics."""
    metrics = evaluate_conversations(sample_conversations)
    
    # Check time metrics
    assert 'avg_duration' in metrics
    assert 'max_duration' in metrics
    assert 'min_duration' in metrics
    
    # Check metric values
    assert metrics['avg_duration'] > 0
    assert metrics['max_duration'] >= metrics['avg_duration']
    assert metrics['min_duration'] <= metrics['avg_duration']

def test_evaluate_conversations_participant_metrics(sample_conversations):
    """Test participant-based metrics."""
    metrics = evaluate_conversations(sample_conversations)
    
    # Check participant metrics
    assert 'unique_participants' in metrics
    assert 'participant_overlap' in metrics
    
    # Check metric values
    assert metrics['unique_participants'] == 4  # user1, user2, user3, user4
    assert 0 <= metrics['participant_overlap'] <= 1

def test_evaluate_conversations_empty():
    """Test evaluation with empty conversations."""
    with pytest.raises(ValueError):
        evaluate_conversations([])

# Clustering Metrics Tests
def test_calculate_clustering_metrics_basic(sample_data, sample_labels):
    """Test basic clustering metrics calculation."""
    metrics = calculate_clustering_metrics(sample_data, sample_labels)
    
    # Check required metrics
    assert 'silhouette_score' in metrics
    assert 'calinski_harabasz_score' in metrics
    assert 'davies_bouldin_score' in metrics
    
    # Check metric ranges
    assert -1 <= metrics['silhouette_score'] <= 1
    assert metrics['calinski_harabasz_score'] >= 0
    assert metrics['davies_bouldin_score'] >= 0

def test_calculate_clustering_metrics_perfect(sample_data):
    """Test metrics with perfect clustering."""
    # Create labels where each point is in its own cluster
    perfect_labels = np.arange(len(sample_data))
    metrics = calculate_clustering_metrics(sample_data, perfect_labels)
    
    # Perfect clustering should have high silhouette score
    assert metrics['silhouette_score'] > 0.8

def test_calculate_clustering_metrics_noise():
    """Test metrics with noise points."""
    # Create data with noise
    noisy_data = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.1],
        [5.0, 5.0],  # Noise point
        [-5.0, -5.0]  # Noise point
    ])
    
    # Labels with noise points marked as -1
    noisy_labels = np.array([0, 0, 1, 1, -1, -1])
    
    metrics = calculate_clustering_metrics(noisy_data, noisy_labels)
    assert 'noise_ratio' in metrics
    assert metrics['noise_ratio'] == 2/6  # 2 noise points out of 6 total

def test_calculate_clustering_metrics_empty():
    """Test metrics with empty data."""
    with pytest.raises(ValueError):
        calculate_clustering_metrics(np.array([]), np.array([]))

def test_calculate_clustering_metrics_single_cluster():
    """Test metrics with a single cluster."""
    data = np.array([[1, 1], [1.1, 0.9], [0.9, 1.1]])
    labels = np.array([0, 0, 0])
    
    metrics = calculate_clustering_metrics(data, labels)
    assert metrics['num_clusters'] == 1
    assert metrics['silhouette_score'] == 0  # Single cluster has silhouette score of 0

def test_calculate_clustering_metrics_mismatched():
    """Test metrics with mismatched data and labels."""
    data = np.array([[1, 1], [2, 2]])
    labels = np.array([0])
    
    with pytest.raises(ValueError):
        calculate_clustering_metrics(data, labels) 