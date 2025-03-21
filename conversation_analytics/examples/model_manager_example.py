"""
Example script demonstrating the usage of ModelManager.

This script shows how to:
1. Initialize the model manager with DeepSeek v3
2. Process conversations in batches
3. Save and load predictions
4. Monitor model performance
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from conversation_analytics.model_manager import ModelManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_messages: int = 1000) -> pd.DataFrame:
    """
    Generate sample data for testing.
    
    Args:
        n_messages: Number of messages to generate
        
    Returns:
        DataFrame with sample messages
    """
    messages = []
    for i in range(n_messages):
        messages.append({
            'message_id': f"msg_{i}",
            'text': f"Sample message {i}",
            'timestamp': pd.Timestamp.now().isoformat()
        })
    return pd.DataFrame(messages)

def main():
    # Initialize model manager with DeepSeek v3
    model_manager = ModelManager(
        model_name="deepseekv3",
        cache_dir="model_cache",
        confidence_threshold=0.7
    )
    
    # Generate sample data
    logger.info("Generating sample data...")
    messages = generate_sample_data(1000)
    
    # Process conversations
    logger.info("Processing conversations...")
    predictions = model_manager.process_conversations(
        messages,
        batch_size=100
    )
    
    # Save predictions
    logger.info("Saving predictions...")
    output_file = model_manager.save_predictions(
        predictions,
        output_dir="predictions"
    )
    logger.info(f"Predictions saved to: {output_file}")
    
    # Load predictions back
    logger.info("Loading predictions...")
    loaded_predictions = model_manager.load_predictions(output_file)
    
    # Get model statistics
    stats = model_manager.get_model_stats()
    logger.info("\nModel Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Print prediction summary
    logger.info("\nPrediction Summary:")
    logger.info(f"  Total messages processed: {len(predictions)}")
    logger.info(f"  Average confidence: {predictions['confidence'].mean():.3f}")
    logger.info(f"  Number of conversations: {len(predictions['conversation_id'].unique())}")
    logger.info(f"  Average messages per conversation: {len(predictions) / len(predictions['conversation_id'].unique()):.2f}")

if __name__ == "__main__":
    main() 