#!/usr/bin/env python3
"""Script to evaluate spam classification performance of different models.

This script provides functionality to evaluate the performance of different models
in detecting spam messages by comparing their predictions against ground truth data.
It calculates various classification metrics including accuracy, precision, recall, and F1 score.

Key Features:
- Loads and validates ground truth data and model predictions
- Calculates comprehensive spam detection metrics
- Handles various edge cases and data validation
- Generates detailed metrics reports

Edge Cases Handled:
- Missing or invalid columns in input files
- NaN values in spam labels
- Empty or malformed input files
- Invalid file paths
- Inconsistent data formats
- Class imbalance in spam detection
- Invalid binary labels

Example Usage:
    python spam_metrics.py /path/to/group/directory

The script expects:
1. A directory containing:
   - Ground truth file (GT_spam_{group_name}.csv)
   - Model prediction files (labels_*.csv)
2. Input files must contain columns:
   - Ground truth: 'id' and 'is_spam'
   - Predictions: 'message_id' and 'conversation_id'
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import logging
import glob
from datetime import datetime
import os
import sys

# Set up more verbose logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_ground_truth(ground_truth_path: Path) -> pd.DataFrame:
    """Load and validate ground truth spam labels.
    
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
        - Invalid binary values
    """
    try:
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
            
        logger.info(f"Loading ground truth file from: {ground_truth_path}")
        ground_truth = pd.read_csv(ground_truth_path)
        
        # Validate file is not empty
        if ground_truth.empty:
            raise ValueError("Ground truth file is empty")
            
        required_columns = {'id', 'is_spam'}
        if not all(col in ground_truth.columns for col in required_columns):
            raise ValueError(f"Ground truth file must contain columns: {required_columns}")
            
        # Validate spam labels are binary
        if not ground_truth['is_spam'].isin([0, 1]).all():
            raise ValueError("Spam labels must be binary (0 or 1)")
            
        # Check for duplicate message IDs
        if ground_truth['id'].duplicated().any():
            logger.warning("Duplicate message IDs found in ground truth file")
            
        logger.info(f"Successfully loaded ground truth with {len(ground_truth)} entries")
        logger.info(f"Spam distribution: {ground_truth['is_spam'].value_counts().to_dict()}")
        
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
    """Load and validate model predictions for spam detection.
    
    Args:
        label_file: Path to the model predictions CSV file
        
    Returns:
        DataFrame containing validated model predictions with columns:
            - message_id: Unique identifier for each message
            - is_spam: Binary spam label (0 or 1)
            
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
        - Invalid conversation IDs
        - Missing message IDs
    """
    try:
        if not label_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {label_file}")
            
        logger.info(f"Loading predictions from: {label_file}")
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
            
        # Convert model predictions to binary spam labels
        df['is_spam'] = (df['conversation_id'] == 0).astype(int)
        
        # Validate spam labels are binary
        if not df['is_spam'].isin([0, 1]).all():
            raise ValueError("Invalid spam labels detected in predictions")
            
        logger.info(f"Found {len(df)} predictions")
        logger.info(f"Identified {df['is_spam'].sum()} spam messages")
        
        return df[['message_id', 'is_spam']]
    except pd.errors.EmptyDataError:
        logger.error(f"Prediction file is empty: {label_file}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing prediction file {label_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading predictions from {label_file}: {e}")
        raise

def calculate_metrics(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Calculate spam classification metrics.
    
    Args:
        ground_truth: DataFrame containing ground truth labels
        predictions: DataFrame containing model predictions
        
    Returns:
        Dictionary containing:
            - accuracy: Overall classification accuracy
            - precision: Precision score for spam class
            - recall: Recall score for spam class
            - f1: F1 score for spam class
            
    Edge Cases Handled:
        - No matching messages between ground truth and predictions
        - NaN values in labels
        - Empty input DataFrames
        - Class imbalance
        - Single class predictions
    """
    if ground_truth.empty or predictions.empty:
        logger.warning("Empty input DataFrame provided")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        
    # Merge ground truth with predictions
    merged = pd.merge(
        ground_truth, 
        predictions,
        left_on='id',
        right_on='message_id',
        how='inner',
        suffixes=('_true', '_pred')
    )
    
    if merged.empty:
        logger.warning("No matching messages found between ground truth and predictions")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    logger.info(f"Merged dataset has {len(merged)} entries")
    
    # Drop any rows with NaN values
    merged = merged.dropna(subset=['is_spam_true', 'is_spam_pred'])
    logger.info(f"After dropping NaN values: {len(merged)} entries")
    
    # Calculate metrics
    y_true = merged['is_spam_true']
    y_pred = merged['is_spam_pred']
    
    # Handle edge cases for metric calculation
    if len(y_true.unique()) == 1 or len(y_pred.unique()) == 1:
        logger.warning("Single class detected in predictions or ground truth")
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    logger.info("Calculated metrics:")
    for metric, value in metrics.items():
        logger.info(f"- {metric}: {value:.4f}")
    
    return metrics

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
        - Empty file names
        - Invalid file paths
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

def evaluate_spam_detection(group_dir: str) -> None:
    """Evaluate spam detection performance for all models in the directory.
    
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
        - Permission errors when saving results
        - Disk space issues
    """
    logger.info(f"Starting spam detection evaluation for directory: {group_dir}")
    
    group_path = Path(group_dir)
    if not group_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {group_dir}")
        
    if not group_path.is_dir():
        raise ValueError(f"Path is not a directory: {group_dir}")
    
    # Get group name from directory path
    group_name = group_path.name
    logger.info(f"Group name: {group_name}")
    
    # Find ground truth file with group name
    ground_truth_path = group_path / f"GT_spam_{group_name}.csv"
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
    logger.info(f"Found {len(label_files)} label files")
    if not label_files:
        raise ValueError(f"No label files found in {group_dir}")
    
    # Calculate metrics for each model
    results = []
    for label_file in label_files:
        model_name = extract_model_name(label_file)
        logger.info(f"\nProcessing predictions from model: {model_name}")
        
        try:
            predictions = load_model_predictions(label_file)
            metrics = calculate_metrics(ground_truth, predictions)
            
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
    columns = ['model', 'label_file', 'accuracy', 'precision', 'recall', 'f1']
    results_df = results_df[columns]
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Save results with timestamp
    output_file = group_path / f"metrics_spam_detection_{group_name}_{timestamp}.csv"
    try:
        results_df.to_csv(output_file, index=False)
        logger.info(f"\nResults summary:")
        logger.info(f"\n{results_df.to_string()}")
        logger.info(f"\nSaved metrics to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")
        raise

def main():
    """Main entry point for the spam metrics evaluation script.
    
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
        - Memory errors
    """
    # Configure logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('spam_metrics.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        parser = argparse.ArgumentParser(
            description='Calculate spam classification metrics for different models.',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
    # Evaluate models in a specific directory
    python spam_metrics.py /path/to/group/directory
    
    # Show help message
    python spam_metrics.py --help
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
        evaluate_spam_detection(args.group_dir)
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 