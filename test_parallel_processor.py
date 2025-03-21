"""Unit tests for the parallel processor module."""

import unittest
import time
from typing import List
from parallel_processor import ParallelProcessor, ProcessingResult

def mock_processor(items: List[str]) -> List[str]:
    """Mock processing function that simulates work."""
    time.sleep(0.1)  # Simulate some work
    return [f"processed_{item}" for item in items]

def failing_processor(items: List[str]) -> List[str]:
    """Mock processing function that always fails."""
    raise ValueError("Simulated processing error")

class TestParallelProcessor(unittest.TestCase):
    """Test cases for ParallelProcessor class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.processor = ParallelProcessor(
            max_workers=3,
            batch_size=5,
            timeout=10
        )
        self.test_items = [f"item_{i}" for i in range(15)]
        
    def tearDown(self):
        """Clean up after each test."""
        self.processor.shutdown()
        
    def test_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.max_workers, 3)
        self.assertEqual(self.processor.batch_size, 5)
        self.assertEqual(self.processor.timeout, 10)
        
    def test_empty_batch(self):
        """Test processing empty batch."""
        results = self.processor.process_batch([], mock_processor)
        self.assertEqual(len(results), 0)
        
    def test_successful_processing(self):
        """Test successful batch processing."""
        results = self.processor.process_batch(self.test_items, mock_processor)
        
        # Check results
        self.assertEqual(len(results), 3)  # 3 chunks
        for result in results:
            self.assertTrue(result.success)
            self.assertIsNone(result.error)
            self.assertGreater(result.processing_time, 0)
            
        # Check stats
        stats = self.processor.get_stats()
        self.assertEqual(stats['processed'], 15)
        self.assertEqual(stats['errors'], 0)
        self.assertGreater(stats['total_time'], 0)
        
    def test_failed_processing(self):
        """Test handling of processing failures."""
        results = self.processor.process_batch(self.test_items, failing_processor)
        
        # Check results
        self.assertEqual(len(results), 3)  # 3 chunks
        for result in results:
            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)
            self.assertIn("Simulated processing error", result.error)
            
        # Check stats
        stats = self.processor.get_stats()
        self.assertEqual(stats['errors'], 3)  # One error per chunk
        
    def test_stats_reset(self):
        """Test statistics reset functionality."""
        # Process some items
        self.processor.process_batch(self.test_items, mock_processor)
        
        # Check stats before reset
        stats_before = self.processor.get_stats()
        self.assertGreater(stats_before['processed'], 0)
        
        # Reset stats
        self.processor.reset_stats()
        
        # Check stats after reset
        stats_after = self.processor.get_stats()
        self.assertEqual(stats_after['processed'], 0)
        self.assertEqual(stats_after['errors'], 0)
        self.assertEqual(stats_after['total_time'], 0)
        
    def test_chunk_splitting(self):
        """Test correct splitting of items into chunks."""
        chunks = self.processor._split_into_chunks(self.test_items)
        
        # Should have 3 chunks of 5 items each
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 5)
        self.assertEqual(len(chunks[1]), 5)
        self.assertEqual(len(chunks[2]), 5)
        
    def test_timeout_handling(self):
        """Test handling of timeout during processing."""
        def slow_processor(items):
            time.sleep(2)  # Longer than timeout
            return items
            
        processor = ParallelProcessor(timeout=1)  # Short timeout
        results = processor.process_batch(self.test_items, slow_processor)
        
        # Should return empty list on timeout
        self.assertEqual(len(results), 0)
        
if __name__ == '__main__':
    unittest.main() 