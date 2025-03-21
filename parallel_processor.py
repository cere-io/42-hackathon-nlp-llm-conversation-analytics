"""
Parallel processing module for efficient message batch processing.

This module provides parallel processing capabilities for handling large batches
of messages efficiently using thread pools.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any, Dict, Optional
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Container for processing results and metadata."""
    data: Any
    processing_time: float
    success: bool
    error: Optional[str] = None

class ParallelProcessor:
    """
    Handles parallel processing of message batches using thread pools.
    
    Attributes:
        max_workers (int): Maximum number of worker threads
        batch_size (int): Size of each processing batch
        timeout (int): Timeout in seconds for each batch
    """
    
    def __init__(self, max_workers: int = 5, batch_size: int = 100, timeout: int = 30):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of concurrent workers
            batch_size: Number of items to process in each batch
            timeout: Maximum time in seconds to wait for a batch
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.stats: Dict[str, int] = {
            'processed': 0,
            'errors': 0,
            'total_time': 0
        }
        logger.info(f"Initialized ParallelProcessor with {max_workers} workers")
        
    def process_batch(self, items: List[Any], processor_func) -> List[ProcessingResult]:
        """
        Process a batch of items in parallel.
        
        Args:
            items: List of items to process
            processor_func: Function to process each batch
            
        Returns:
            List of ProcessingResult objects containing results and metadata
        """
        if not items:
            logger.warning("Received empty item list")
            return []
            
        start_time = time.time()
        chunks = self._split_into_chunks(items)
        futures = []
        
        logger.info(f"Processing {len(items)} items in {len(chunks)} chunks")
        
        try:
            # Submit all chunks for processing
            for chunk in chunks:
                future = self.executor.submit(self._process_chunk, chunk, processor_func)
                futures.append(future)
            
            # Collect results as they complete
            results = []
            for future in as_completed(futures, timeout=self.timeout):
                try:
                    result = future.result()
                    results.extend(result)
                    if not result[0].success:  # Check if processing failed
                        self.stats['errors'] += 1
                    else:
                        self.stats['processed'] += len(chunk)
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    self.stats['errors'] += 1
                    results.append(ProcessingResult(
                        data=None,
                        processing_time=time.time() - start_time,
                        success=False,
                        error=str(e)
                    ))
                    
            total_time = time.time() - start_time
            self.stats['total_time'] += total_time
            
            logger.info(f"Batch processing completed in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return []
            
    def _split_into_chunks(self, items: List[Any]) -> List[List[Any]]:
        """Split items into chunks of batch_size."""
        return [items[i:i + self.batch_size] 
                for i in range(0, len(items), self.batch_size)]
                
    def _process_chunk(self, chunk: List[Any], processor_func) -> List[ProcessingResult]:
        """Process a single chunk and return results with metadata."""
        start_time = time.time()
        try:
            result = processor_func(chunk)
            processing_time = time.time() - start_time
            return [ProcessingResult(
                data=result,
                processing_time=processing_time,
                success=True
            )]
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Chunk processing error: {str(e)}")
            return [ProcessingResult(
                data=None,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )]
            
    def get_stats(self) -> Dict[str, int]:
        """Return current processing statistics."""
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'processed': 0,
            'errors': 0,
            'total_time': 0
        }
        
    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)
        logger.info("ParallelProcessor shutdown complete") 