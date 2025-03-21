"""
Tests for all processor classes including text vectorizer and parallel processor.
"""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool

from src.processors.text_vectorizer import TextVectorizer
from src.processors.parallel_processor import ParallelProcessor

# Fixtures
@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "Hello world",
        "Hello there",
        "Good morning world",
        "Good morning there",
        "Have a nice day"
    ]

@pytest.fixture
def sample_data():
    """Create sample data for parallel processing."""
    return [
        {"id": 1, "text": "Hello world"},
        {"id": 2, "text": "Hello there"},
        {"id": 3, "text": "Good morning"},
        {"id": 4, "text": "Good night"},
        {"id": 5, "text": "Goodbye world"}
    ]

# Text Vectorizer Tests
def test_text_vectorizer_initialization():
    """Test text vectorizer initialization."""
    vectorizer = TextVectorizer()
    assert isinstance(vectorizer.vectorizer, TfidfVectorizer)
    assert vectorizer.cache_size == 1000
    assert vectorizer.cache_expiry == 3600

def test_text_vectorizer_fit(sample_texts):
    """Test text vectorizer fitting."""
    vectorizer = TextVectorizer()
    vectorizer.fit(sample_texts)
    
    # Check that vocabulary was built
    assert len(vectorizer.vectorizer.vocabulary_) > 0
    assert "hello" in vectorizer.vectorizer.vocabulary_
    assert "world" in vectorizer.vectorizer.vocabulary_

def test_text_vectorizer_transform(sample_texts):
    """Test text vector transformation."""
    vectorizer = TextVectorizer()
    vectorizer.fit(sample_texts)
    vectors = vectorizer.transform(sample_texts)
    
    # Check vector dimensions
    assert vectors.shape[0] == len(sample_texts)
    assert vectors.shape[1] > 0
    
    # Check that vectors are normalized
    for i in range(vectors.shape[0]):
        norm = np.linalg.norm(vectors[i].toarray())
        assert np.isclose(norm, 1.0)

def test_text_vectorizer_cache(sample_texts):
    """Test text vectorizer caching."""
    vectorizer = TextVectorizer()
    vectorizer.fit(sample_texts)
    
    # First transformation should not be cached
    vectors1 = vectorizer.transform(sample_texts)
    
    # Second transformation should be cached
    vectors2 = vectorizer.transform(sample_texts)
    assert np.array_equal(vectors1.toarray(), vectors2.toarray())
    
    # Check cache size limit
    for i in range(vectorizer.cache_size + 1):
        vectorizer.transform([f"Text {i}"])
    assert len(vectorizer._cache) <= vectorizer.cache_size

# Parallel Processor Tests
def test_parallel_processor_initialization():
    """Test parallel processor initialization."""
    processor = ParallelProcessor()
    assert processor.n_jobs == -1  # Default to all cores
    assert processor.chunk_size == 1000
    assert processor.timeout == 3600

def test_parallel_processor_map(sample_data):
    """Test parallel processing with map."""
    def process_item(item):
        return {"id": item["id"], "length": len(item["text"])}
    
    processor = ParallelProcessor()
    results = processor.map(process_item, sample_data)
    
    assert len(results) == len(sample_data)
    for result, data in zip(results, sample_data):
        assert result["id"] == data["id"]
        assert result["length"] == len(data["text"])

def test_parallel_processor_batch_processing(sample_data):
    """Test batch processing."""
    def process_batch(batch):
        return [len(item["text"]) for item in batch]
    
    processor = ParallelProcessor(chunk_size=2)
    results = processor.process_batches(process_batch, sample_data)
    
    assert len(results) == len(sample_data)
    for result, data in zip(results, sample_data):
        assert result == len(data["text"])

def test_parallel_processor_error_handling():
    """Test error handling in parallel processing."""
    def failing_function(item):
        raise ValueError(f"Error processing item {item}")
    
    processor = ParallelProcessor()
    with pytest.raises(ValueError):
        processor.map(failing_function, [1, 2, 3])

def test_parallel_processor_timeout():
    """Test timeout handling."""
    def slow_function(item):
        import time
        time.sleep(2)
        return item
    
    processor = ParallelProcessor(timeout=1)
    with pytest.raises(TimeoutError):
        processor.map(slow_function, [1, 2, 3])

# Error Handling Tests
def test_processors_empty_data():
    """Test handling of empty data."""
    vectorizer = TextVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit([])
    
    processor = ParallelProcessor()
    with pytest.raises(ValueError):
        processor.map(lambda x: x, [])

def test_processors_invalid_data():
    """Test handling of invalid data."""
    vectorizer = TextVectorizer()
    with pytest.raises(ValueError):
        vectorizer.fit([1, 2, 3])  # Non-string input
    
    processor = ParallelProcessor()
    with pytest.raises(ValueError):
        processor.map("not a function", [1, 2, 3])

# Persistence Tests
def test_processors_persistence(tmp_path, sample_texts):
    """Test processor persistence."""
    vectorizer = TextVectorizer()
    vectorizer.fit(sample_texts)
    
    # Save vectorizer
    save_path = tmp_path / "vectorizer.pkl"
    vectorizer.save(save_path)
    
    # Load vectorizer
    loaded_vectorizer = TextVectorizer.load(save_path)
    vectors1 = vectorizer.transform(sample_texts)
    vectors2 = loaded_vectorizer.transform(sample_texts)
    assert np.array_equal(vectors1.toarray(), vectors2.toarray()) 