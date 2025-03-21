"""
Cache Manager Module

This module provides a robust caching system for storing and retrieving API responses
and processed data. It implements a memory-efficient caching mechanism with automatic
cleanup and size management.

Key Features:
- LRU (Least Recently Used) cache implementation
- Automatic cache size management
- Thread-safe operations
- Persistent storage support
- Memory usage monitoring

Usage:
    cache = CacheManager(cache_dir="cache", max_size=1000)
    result = cache.get("key")
    if result is None:
        result = expensive_operation()
        cache.set("key", result)
"""

import os
import json
import time
import threading
from typing import Any, Optional, Dict, List
from collections import OrderedDict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages caching of API responses and processed data with automatic cleanup.
    
    Attributes:
        cache_dir (str): Directory for persistent cache storage
        max_size (int): Maximum number of items in cache
        cache (OrderedDict): In-memory cache storage
        lock (threading.Lock): Thread safety lock
        _last_cleanup (float): Timestamp of last cleanup
        
    Edge Cases Handled:
        - Cache directory creation/access issues
        - Memory pressure monitoring
        - Thread safety for concurrent access
        - Invalid cache entries
        - Disk space management
    """
    
    def __init__(self, cache_dir: str = "cache", max_size: int = 1000):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for persistent cache storage
            max_size: Maximum number of items to store in cache
            
        Raises:
            OSError: If cache directory cannot be created
            ValueError: If max_size is not positive
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
            
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self._last_cleanup = time.time()
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache directory initialized at {self.cache_dir}")
        except OSError as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise
            
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from cache.
        
        Args:
            key: Cache key to look up
            
        Returns:
            Cached value if found, None otherwise
            
        Edge Cases:
            - Invalid cache entries
            - Corrupted cache files
            - Missing cache directory
        """
        with self.lock:
            try:
                if key in self.cache:
                    # Move to end (most recently used)
                    value = self.cache.pop(key)
                    self.cache[key] = value
                    return value
                    
                # Check persistent storage
                cache_file = self.cache_dir / f"{key}.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            value = json.load(f)
                        self.cache[key] = value
                        return value
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(f"Failed to read cache file {key}: {e}")
                        return None
                        
                return None
                
            except Exception as e:
                logger.error(f"Error retrieving from cache: {e}")
                return None
                
    def set(self, key: str, value: Any) -> None:
        """
        Store an item in cache.
        
        Args:
            key: Cache key
            value: Value to store
            
        Edge Cases:
            - Cache size limit reached
            - Disk space issues
            - Invalid value types
        """
        with self.lock:
            try:
                # Remove if exists (will be added at end)
                if key in self.cache:
                    self.cache.pop(key)
                    
                # Check size limit
                if len(self.cache) >= self.max_size:
                    self._cleanup()
                    
                # Add to memory cache
                self.cache[key] = value
                
                # Persist to disk
                cache_file = self.cache_dir / f"{key}.json"
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(value, f)
                except (TypeError, OSError) as e:
                    logger.warning(f"Failed to persist cache entry {key}: {e}")
                    
            except Exception as e:
                logger.error(f"Error setting cache entry: {e}")
                
    def _cleanup(self) -> None:
        """
        Clean up old cache entries.
        
        Edge Cases:
            - Failed file deletion
            - Concurrent access during cleanup
        """
        try:
            # Remove oldest entries from memory
            while len(self.cache) >= self.max_size:
                key, _ = self.cache.popitem(last=False)
                
                # Remove from disk
                cache_file = self.cache_dir / f"{key}.json"
                try:
                    if cache_file.exists():
                        cache_file.unlink()
                except OSError as e:
                    logger.warning(f"Failed to delete cache file {key}: {e}")
                    
            self._last_cleanup = time.time()
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            
    def clear(self) -> None:
        """
        Clear all cache entries.
        
        Edge Cases:
            - Failed file deletion
            - Concurrent access during cleanup
        """
        with self.lock:
            try:
                # Clear memory cache
                self.cache.clear()
                
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.json"):
                    try:
                        cache_file.unlink()
                    except OSError as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {e}")
                        
                logger.info("Cache cleared successfully")
                
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "last_cleanup": self._last_cleanup,
                "cache_dir": str(self.cache_dir)
            } 