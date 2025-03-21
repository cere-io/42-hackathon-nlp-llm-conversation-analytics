"""
Unit tests for the CacheManager class.

This module contains comprehensive tests for all CacheManager functionality,
including edge cases and error conditions.
"""

import unittest
import tempfile
import shutil
import os
import json
from pathlib import Path
from cache_manager import CacheManager

class TestCacheManager(unittest.TestCase):
    """Test suite for CacheManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.temp_dir, max_size=3)
        
    def tearDown(self):
        """Clean up test environment after each test."""
        shutil.rmtree(self.temp_dir)
        
    def test_initialization(self):
        """Test cache manager initialization."""
        self.assertEqual(self.cache.max_size, 3)
        self.assertEqual(len(self.cache.cache), 0)
        self.assertTrue(os.path.exists(self.temp_dir))
        
    def test_set_and_get(self):
        """Test basic set and get operations."""
        # Test setting and getting a value
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Test getting non-existent key
        self.assertIsNone(self.cache.get("nonexistent"))
        
    def test_lru_behavior(self):
        """Test Least Recently Used (LRU) behavior."""
        # Fill cache to capacity
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        # Access key1 to make it most recent
        self.cache.get("key1")
        
        # Add new key, should remove least recently used (key2)
        self.cache.set("key4", "value4")
        self.assertIsNone(self.cache.get("key2"))
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key3"), "value3")
        self.assertEqual(self.cache.get("key4"), "value4")
        
    def test_persistence(self):
        """Test cache persistence to disk."""
        # Create new cache instance
        cache1 = CacheManager(cache_dir=self.temp_dir, max_size=3)
        cache1.set("key1", "value1")
        
        # Create another instance pointing to same directory
        cache2 = CacheManager(cache_dir=self.temp_dir, max_size=3)
        self.assertEqual(cache2.get("key1"), "value1")
        
    def test_clear(self):
        """Test cache clearing functionality."""
        # Add some values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Clear cache
        self.cache.clear()
        
        # Verify cache is empty
        self.assertEqual(len(self.cache.cache), 0)
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))
        
        # Verify files are removed
        cache_files = list(Path(self.temp_dir).glob("*.json"))
        self.assertEqual(len(cache_files), 0)
        
    def test_invalid_values(self):
        """Test handling of invalid values."""
        # Test with non-serializable value
        class NonSerializable:
            pass
            
        self.cache.set("key1", NonSerializable())
        self.assertIsNone(self.cache.get("key1"))
        
    def test_concurrent_access(self):
        """Test thread safety of cache operations."""
        import threading
        
        def worker():
            for i in range(100):
                self.cache.set(f"key{i}", f"value{i}")
                self.cache.get(f"key{i}")
                
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
            
        # Verify cache is in valid state
        self.assertLessEqual(len(self.cache.cache), self.cache.max_size)
        
    def test_stats(self):
        """Test cache statistics reporting."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        stats = self.cache.get_stats()
        self.assertEqual(stats["size"], 2)
        self.assertEqual(stats["max_size"], 3)
        self.assertIn("last_cleanup", stats)
        self.assertEqual(stats["cache_dir"], self.temp_dir)
        
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test invalid max_size
        with self.assertRaises(ValueError):
            CacheManager(cache_dir=self.temp_dir, max_size=0)
            
        # Test invalid cache directory
        with self.assertRaises(OSError):
            CacheManager(cache_dir="/invalid/path/that/does/not/exist")
            
        # Test corrupted cache file
        cache_file = Path(self.temp_dir) / "corrupted.json"
        with open(cache_file, "w") as f:
            f.write("invalid json")
            
        self.assertIsNone(self.cache.get("corrupted"))
        
if __name__ == '__main__':
    unittest.main() 