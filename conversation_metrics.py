#!/usr/bin/env python3
"""Script to evaluate conversation clustering performance of different models.

This script provides functionality to evaluate the performance of different models
in clustering conversations by comparing their predictions against ground truth data.
It calculates the Adjusted Rand Index (ARI) as the main metric for clustering quality.

Key Features:
- Loads and validates ground truth data and model predictions
- Calculates ARI for conversation clustering
- Handles various edge cases and data validation
- Generates detailed metrics reports

Edge Cases Handled:
- Missing or invalid columns in input files
- NaN values in conversation IDs
- Empty or malformed input files
- Invalid file paths
- Inconsistent data formats

Example Usage:
    python conversation_metrics.py /path/to/group/directory

The script expects:
1. A directory containing:
   - Ground truth file (GT_conversations_{group_name}.csv)
   - Model prediction files (labels_*.csv)
2. Input files must contain columns: 'message_id' and 'conversation_id'
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
import argparse
import logging
import glob
from datetime import datetime
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ground_truth(ground_truth_path: Path) -> pd.DataFrame:
    """Load and validate ground truth conversation labels.
    
    Args:
        ground_truth_path: Path to the ground truth CSV file
        
    Returns:
        DataFrame containing validated ground truth labels
        
    Raises:
        ValueError: If required columns are missing or file is invalid
        FileNotFoundError: If the ground truth file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file is not a valid CSV
        
    Edge Cases Handled:
        - Empty files
        - Missing required columns
        - Invalid data types
        - Duplicate message IDs
    """
    try:
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
            
        ground_truth = pd.read_csv(ground_truth_path)
        
        # Validate file is not empty
        if ground_truth.empty:
            raise ValueError("Ground truth file is empty")
            
        required_columns = {'id', 'conv_id'}
        if not all(col in ground_truth.columns for col in required_columns):
            raise ValueError(f"Ground truth file must contain columns: {required_columns}")
            
        # Check for duplicate message IDs
        if ground_truth['id'].duplicated().any():
            logger.warning("Duplicate message IDs found in ground truth file")
            
        # Rename columns to match the format we use
        ground_truth = ground_truth.rename(columns={
            'id': 'message_id',
            'conv_id': 'conversation_id'
        })
        
        return ground_truth
    except pd.errors.EmptyDataError:
        logger.error("Ground truth file is empty")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing ground truth file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading ground truth file: {e}")
        raise

def load_model_predictions(label_file: Path) -> pd.DataFrame:
    """Load and validate model predictions for conversation clustering.
    
    Args:
        label_file: Path to the model predictions CSV file
        
    Returns:
        DataFrame containing validated model predictions with columns:
            - message_id: Unique identifier for each message
            - conversation_id: Predicted conversation group
            
    Raises:
        ValueError: If required columns are missing or file is invalid
        FileNotFoundError: If the prediction file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file is not a valid CSV
        
    Edge Cases Handled:
        - Empty files
        - Missing required columns
        - Invalid data types
        - Duplicate message IDs
        - Malformed conversation IDs
    """
    try:
        if not label_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {label_file}")
            
        df = pd.read_csv(label_file)
        
        # Validate file is not empty
        if df.empty:
            raise ValueError(f"Prediction file is empty: {label_file}")
            
        required_columns = {'message_id', 'conversation_id'}
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Label file must contain columns: {required_columns}")
            
        # Check for duplicate message IDs
        if df['message_id'].duplicated().any():
            logger.warning(f"Duplicate message IDs found in prediction file: {label_file}")
            
        # Validate conversation IDs are not NaN
        if df['conversation_id'].isna().any():
            logger.warning(f"NaN values found in conversation IDs: {label_file}")
            
        return df[['message_id', 'conversation_id']]
    except pd.errors.EmptyDataError:
        logger.error(f"Prediction file is empty: {label_file}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing prediction file {label_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading predictions from {label_file}: {e}")
        raise

def calculate_ari(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Calculate Adjusted Rand Index for conversation clustering.
    
    Args:
        ground_truth: DataFrame containing ground truth labels
        predictions: DataFrame containing model predictions
        
    Returns:
        Dictionary containing:
            - ari: Adjusted Rand Index score
            - n_messages: Number of messages used in calculation
            
    Edge Cases Handled:
        - No matching messages between ground truth and predictions
        - NaN values in conversation IDs
        - Empty input DataFrames
        - Single conversation case
    """
    if ground_truth.empty or predictions.empty:
        logger.warning("Empty input DataFrame provided")
        return {'ari': 0.0, 'n_messages': 0}
        
    # Merge ground truth with predictions
    merged = pd.merge(
        ground_truth,
        predictions,
        on='message_id',
        how='inner',
        suffixes=('_true', '_pred')
    )
    
    if merged.empty:
        logger.warning("No matching messages found between ground truth and predictions")
        return {'ari': 0.0, 'n_messages': 0}
    
    # Drop any rows with NaN values
    merged = merged.dropna(subset=['conversation_id_true', 'conversation_id_pred'])
    
    if len(merged) < 2:
        logger.warning("Insufficient data for ARI calculation")
        return {'ari': 0.0, 'n_messages': len(merged)}
    
    # Calculate ARI
    ari = adjusted_rand_score(
        merged['conversation_id_true'],
        merged['conversation_id_pred']
    )
    
    return {
        'ari': ari,
        'n_messages': len(merged)
    }

def extract_model_name(label_file: Path) -> str:
    """Extract model name from label file name.
    
    Args:
        label_file: Path to the label file
        
    Returns:
        String containing the extracted model name
        
    Edge Cases Handled:
        - Files with unexpected naming patterns
        - Files without model name in pattern
        - Files with special characters
    """
    try:
        # Expected format: labels_YYYYMMDD_modelname_groupname.csv
        parts = label_file.stem.split('_')
        
        if len(parts) < 3:
            logger.warning(f"Unexpected file name format: {label_file.name}")
            return label_file.stem
            
        # Clean model name of any special characters
        model_name = parts[2].strip().lower()
        return model_name
    except Exception as e:
        logger.warning(f"Error extracting model name from {label_file.name}: {e}")
        return label_file.stem

def evaluate_conversation_clustering(group_dir: str) -> None:
    """Evaluate conversation clustering performance for all models in the directory.
    
    Args:
        group_dir: Path to directory containing label files and ground truth
        
    Raises:
        ValueError: If directory is invalid or required files are missing
        FileNotFoundError: If directory doesn't exist
        RuntimeError: If no results could be calculated
        
    Edge Cases Handled:
        - Empty directories
        - Missing ground truth file
        - No label files found
        - Invalid file formats
        - Processing errors for individual models
        - No valid results after processing
    """
    group_path = Path(group_dir)
    if not group_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {group_dir}")
        
    if not group_path.is_dir():
        raise ValueError(f"Path is not a directory: {group_dir}")
    
    # Get group name from directory path
    group_name = group_path.name
    
    # Find ground truth file with group name
    ground_truth_path = group_path / f"GT_conversations_{group_name}.csv"
    if not ground_truth_path.exists():
        raise ValueError(f"Ground truth file not found: {ground_truth_path}")
    
    # Load ground truth
    logger.info(f"Loading ground truth from {ground_truth_path}")
    try:
        ground_truth = load_ground_truth(ground_truth_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ground truth: {e}")
    
    # Find all label files
    label_files = list(group_path.glob("labels_*.csv"))
    if not label_files:
        raise ValueError(f"No label files found in {group_dir}")
    
    # Calculate metrics for each model
    results = []
    for label_file in label_files:
        model_name = extract_model_name(label_file)
        logger.info(f"Processing predictions from model: {model_name}")
        
        try:
            predictions = load_model_predictions(label_file)
            metrics = calculate_ari(ground_truth, predictions)
            
            # Add model name and file info to metrics
            metrics['model'] = model_name
            metrics['label_file'] = label_file.name
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error processing {label_file}: {e}")
            continue
    
    if not results:
        raise RuntimeError("No results could be calculated from any label file")
    
    # Create results DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put model first
    columns = ['model', 'label_file', 'ari', 'n_messages']
    results_df = results_df[columns]
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Save results with timestamp
    output_file = group_path / f"metrics_conversations_{group_name}_{timestamp}.csv"
    try:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved metrics to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")
        raise

def main():
    """Main entry point for the conversation metrics evaluation script.
    
    This function:
    1. Sets up logging configuration
    2. Parses command line arguments
    3. Executes the evaluation process
    4. Handles any errors that occur during execution
    
    Edge Cases Handled:
        - Invalid command line arguments
        - Permission errors
        - System errors
        - Keyboard interrupts
    """
    # Configure logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('conversation_metrics.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        parser = argparse.ArgumentParser(
            description='Calculate conversation clustering metrics (ARI) for different models.',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
    # Evaluate models in a specific directory
    python conversation_metrics.py /path/to/group/directory
    
    # Show help message
    python conversation_metrics.py --help
            """
        )
        parser.add_argument(
            'group_dir',
            help='Path to directory containing label files and ground truth'
        )
        
        args = parser.parse_args()
        
        # Validate input directory
        if not os.path.exists(args.group_dir):
            logger.error(f"Directory does not exist: {args.group_dir}")
            sys.exit(1)
            
        if not os.path.isdir(args.group_dir):
            logger.error(f"Path is not a directory: {args.group_dir}")
            sys.exit(1)
            
        # Execute evaluation
        evaluate_conversation_clustering(args.group_dir)
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 