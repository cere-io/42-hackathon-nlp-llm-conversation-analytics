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

import os
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
import argparse
import logging
import glob
from datetime import datetime
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_groupings(ground_truth_df: pd.DataFrame, prediction_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate grouping quality using multiple metrics.
    
    Args:
        ground_truth_df: DataFrame with ground truth labels
        prediction_df: DataFrame with predicted labels
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    # Rename columns to match
    gt_df = ground_truth_df.copy()
    pred_df = prediction_df.copy()
    
    # Rename ground truth columns
    gt_df = gt_df.rename(columns={
        'id': 'message_id',
        'conv_id': 'conversation_id'
    })
    
    # Ensure we have the required columns
    required_cols = ['message_id', 'conversation_id']
    for df, name in [(gt_df, 'ground truth'), (pred_df, 'predictions')]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {name}: {missing}")
    
    # Merge datasets
    merged_df = pd.merge(gt_df, pred_df, on='message_id', how='inner')
    
    metrics = {}
    
    # Calculate ARI
    metrics['ari'] = adjusted_rand_score(
        merged_df['conversation_id_x'],
        merged_df['conversation_id_y']
    )
    
    # Calculate group statistics
    metrics['n_ground_truth_groups'] = merged_df['conversation_id_x'].nunique()
    metrics['n_predicted_groups'] = merged_df['conversation_id_y'].nunique()
    metrics['avg_group_size_gt'] = len(merged_df) / metrics['n_ground_truth_groups']
    metrics['avg_group_size_pred'] = len(merged_df) / metrics['n_predicted_groups']
    
    # Get timestamp
    if 'timestamp' in prediction_df.columns:
        metrics['timestamp'] = prediction_df['timestamp'].iloc[0]
    else:
        # Extract timestamp from filename
        timestamp_str = prediction_df.name.split('_')[1]
        metrics['timestamp'] = pd.to_datetime(timestamp_str, format='%Y%m%d')
    
    return metrics

def visualize_results(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate visualizations of grouping results.
    
    Args:
        metrics_df: DataFrame containing evaluation metrics
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot ARI scores over time
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['timestamp'], metrics_df['ari'], marker='o')
    plt.title('ARI Score Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('ARI Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ari_over_time.png'))
    plt.close()
    
    # Plot coherence metrics
    coherence_metrics = ['temporal_coherence_pred', 'semantic_coherence_pred']
    plt.figure(figsize=(10, 6))
    for metric in coherence_metrics:
        plt.plot(metrics_df['timestamp'], metrics_df[metric], 
                marker='o', label=metric.replace('_', ' ').title())
    plt.title('Coherence Metrics Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Coherence Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coherence_metrics.png'))
    plt.close()
    
    # Generate correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation = metrics_df[[
        'ari', 'temporal_coherence_pred', 'semantic_coherence_pred',
        'n_predicted_groups', 'avg_group_size_pred'
    ]].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation between Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_correlations.png'))
    plt.close()

def generate_report(metrics_df: pd.DataFrame, output_file: str) -> None:
    """
    Generate a detailed analysis report.
    
    Args:
        metrics_df: DataFrame containing evaluation metrics
        output_file: Path to save the report
    """
    with open(output_file, 'w') as f:
        f.write("Conversation Detection Analysis Report\n")
        f.write("===================================\n\n")
        
        # Overall statistics
        f.write("Overall Performance:\n")
        f.write(f"Average ARI Score: {metrics_df['ari'].mean():.4f}\n")
        f.write(f"Best ARI Score: {metrics_df['ari'].max():.4f}\n")
        f.write(f"Worst ARI Score: {metrics_df['ari'].min():.4f}\n\n")
        
        # Group statistics
        f.write("Group Statistics:\n")
        f.write(f"Average Number of Groups: {metrics_df['n_predicted_groups'].mean():.2f}\n")
        f.write(f"Average Group Size: {metrics_df['avg_group_size_pred'].mean():.2f}\n\n")
        
        # Best performing models
        f.write("Top 3 Models:\n")
        top_models = metrics_df.nlargest(3, 'ari')
        for _, model in top_models.iterrows():
            f.write(f"Model {model['model']}: ARI = {model['ari']:.4f}\n")

def main():
    """Main function to run metrics evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate conversation detection results")
    parser.add_argument("group_dir", help="Directory containing ground truth and predictions")
    parser.add_argument("--output-dir", default="results", help="Directory for output files")
    args = parser.parse_args()
    
    try:
        # Load ground truth
        gt_file = os.path.join(args.group_dir, "GT_conversations_thisiscere.csv")
        gt_df = pd.read_csv(gt_file)
        logger.info(f"Loaded ground truth from {gt_file}")
        
        # Process all prediction files
        pred_files = glob.glob(os.path.join(args.group_dir, "labels_*.csv"))
        all_metrics = []
        
        for pred_file in pred_files:
            model_id = os.path.basename(pred_file).split('_')[1]
            logger.info(f"Processing predictions from model: {model_id}")
            
            try:
                pred_df = pd.read_csv(pred_file)
                pred_df.name = os.path.basename(pred_file)  # Store filename for timestamp extraction
                
                # Check for duplicates
                if pred_df['message_id'].duplicated().any():
                    logger.warning(f"Duplicate message IDs found in prediction file: {pred_file}")
                    pred_df = pred_df.drop_duplicates('message_id')
                
                # Calculate metrics
                metrics = evaluate_groupings(gt_df, pred_df)
                metrics['model'] = model_id
                metrics['label_file'] = os.path.basename(pred_file)
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error processing {pred_file}: {str(e)}")
                continue
        
        if not all_metrics:
            logger.error("No metrics were calculated successfully")
            return 1
        
        # Combine all metrics
        metrics_df = pd.DataFrame(all_metrics)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(
            args.group_dir,
            f"metrics_conversations_thisiscere_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        )
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Generate report
        report_file = os.path.join(args.output_dir, 'analysis_report.txt')
        generate_report(metrics_df, report_file)
        logger.info(f"Generated analysis report at {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 