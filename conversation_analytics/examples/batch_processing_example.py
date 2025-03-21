import pandas as pd
import numpy as np
from pathlib import Path
import logging
from conversation_analytics.batch_processor import BatchProcessor
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_messages: int = 10000) -> pd.DataFrame:
    """Generate sample message data for testing."""
    # Random text templates
    templates = [
        "I really {feeling} this {product}! It's {adjective}.",
        "The customer service was {quality}. {detail}",
        "Just bought the new {product} and it's {adjective}!",
        "URGENT: Click here to claim your FREE {product}! Limited time offer!",
        "Having issues with {product}. {detail}",
        "Great experience with {product}. {detail}"
    ]
    
    feelings = ['love', 'like', 'enjoy', 'appreciate']
    products = ['phone', 'laptop', 'tablet', 'smartwatch', 'headphones']
    adjectives = ['amazing', 'fantastic', 'great', 'disappointing', 'terrible']
    qualities = ['excellent', 'good', 'poor', 'terrible']
    details = [
        "Would recommend!",
        "Never buying again.",
        "Best purchase ever.",
        "Waste of money.",
        "Very satisfied."
    ]
    
    # Generate random messages
    messages = []
    for _ in range(n_messages):
        template = np.random.choice(templates)
        message = template.format(
            feeling=np.random.choice(feelings),
            product=np.random.choice(products),
            adjective=np.random.choice(adjectives),
            quality=np.random.choice(qualities),
            detail=np.random.choice(details)
        )
        messages.append(message)
    
    # Create DataFrame
    return pd.DataFrame({
        'message_id': range(n_messages),
        'text': messages,
        'timestamp': pd.date_range(start='2024-01-01', periods=n_messages, freq='1min')
    })

def main():
    # Generate sample data
    logger.info("Generating sample data...")
    df = generate_sample_data(n_messages=10000)
    
    # Initialize batch processor with caching
    cache_dir = Path("cache/batch_processor")
    processor = BatchProcessor(
        batch_size=500,
        max_memory_mb=1024,
        n_workers=4,
        cache_dir=str(cache_dir)
    )
    
    # Process messages in two runs to demonstrate caching
    logger.info("\nFirst run (no cache):")
    start_time = time.time()
    processed_batches = []
    for batch in processor.process_messages(df):
        processed_batches.append(batch)
    first_run_time = time.time() - start_time
    
    # Display first run metrics
    metrics = processor.get_performance_metrics()
    logger.info(f"\nFirst run completed in {first_run_time:.2f} seconds")
    
    # Process again to demonstrate cache
    logger.info("\nSecond run (with cache):")
    start_time = time.time()
    cached_batches = []
    for batch in processor.process_messages(df):
        cached_batches.append(batch)
    second_run_time = time.time() - start_time
    
    # Compare runs
    logger.info("\nPerformance Comparison:")
    logger.info(f"First run time: {first_run_time:.2f}s")
    logger.info(f"Second run time: {second_run_time:.2f}s")
    logger.info(f"Speed improvement: {(first_run_time/second_run_time):.1f}x faster with cache")
    
    # Analyze results
    total_messages = sum(len(batch) for batch in processed_batches)
    spam_messages = sum(batch['is_spam'].sum() for batch in processed_batches)
    
    logger.info("\nProcessing Results:")
    logger.info(f"Total messages processed: {total_messages}")
    logger.info(f"Spam messages detected: {spam_messages} ({spam_messages/total_messages*100:.1f}%)")
    
    # Sample of processed messages
    logger.info("\nSample of processed messages:")
    sample_batch = processed_batches[0].head(3)
    for _, row in sample_batch.iterrows():
        logger.info("\nMessage:")
        logger.info(f"Original: {row['text']}")
        logger.info(f"Cleaned: {row['cleaned_text']}")
        logger.info(f"Tokens: {row['tokens'][:5]}...")
        logger.info(f"Is spam: {row['is_spam']}")
        logger.info(f"Vector shape: {len(row['vector'])}")

if __name__ == "__main__":
    main() 