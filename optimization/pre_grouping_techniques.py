#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Pre-grouping Techniques for Conversation Detection
--------------------------------------------------------
This script implements advanced pre-grouping techniques with
sophisticated data analysis, metrics, and visualization capabilities.
"""

import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MessageAnalyzer:
    """Class to handle message analysis and grouping."""
    
    def __init__(self, input_file: str):
        """Initialize analyzer with input file."""
        self.input_file = input_file
        self.messages = None
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
    def load_messages(self) -> Optional[pd.DataFrame]:
        """Load and preprocess messages from CSV file."""
        try:
            df = pd.read_csv(self.input_file)
            logger.info(f"Loaded {len(df)} messages from {self.input_file}")
            
            # Data validation
            required_columns = ['id', 'text', 'timestamp', 'username']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Data cleaning
            df['text'] = df['text'].fillna('')
            df['username'] = df['username'].fillna('unknown_user')
            
            # Convert timestamps
            df['datetime'] = pd.to_datetime(df['timestamp'])
            
            # Basic statistics
            self._log_data_statistics(df)
            
            self.messages = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading messages: {str(e)}")
            return None
            
    def _log_data_statistics(self, df: pd.DataFrame) -> None:
        """Log basic statistics about the dataset."""
        stats = {
            'total_messages': len(df),
            'unique_users': df['username'].nunique(),
            'time_span': f"{df['datetime'].min()} to {df['datetime'].max()}",
            'avg_message_length': df['text'].str.len().mean()
        }
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

    def time_based_grouping(self, time_window_minutes: int = 30) -> pd.DataFrame:
        """Enhanced time-based grouping with adaptive window."""
        messages = self.messages.copy()
        
        # Calculate message density
        time_diffs = messages['datetime'].diff().dt.total_seconds() / 60
        avg_time_diff = time_diffs.mean()
        
        # Adjust window based on message density
        adaptive_window = min(max(avg_time_diff * 2, 5), time_window_minutes)
        logger.info(f"Using adaptive time window of {adaptive_window:.2f} minutes")
        
        current_group = 1
        messages['time_group'] = 0
        
        for i in range(len(messages)):
            if i == 0:
                messages.loc[messages.index[i], 'time_group'] = current_group
            else:
                time_diff = messages.iloc[i]['datetime'] - messages.iloc[i-1]['datetime']
                if time_diff.total_seconds() / 60 <= adaptive_window:
                    messages.loc[messages.index[i], 'time_group'] = current_group
                else:
                    current_group += 1
                    messages.loc[messages.index[i], 'time_group'] = current_group
        
        self._visualize_time_groups(messages)
        return messages

    def _visualize_time_groups(self, messages: pd.DataFrame) -> None:
        """Create visualization of time-based groups."""
        plt.figure(figsize=(12, 6))
        for group in messages['time_group'].unique():
            group_msgs = messages[messages['time_group'] == group]
            plt.scatter(group_msgs['datetime'], [group] * len(group_msgs), label=f'Group {group}')
        
        plt.title('Time-based Message Groups')
        plt.xlabel('Time')
        plt.ylabel('Group')
        plt.savefig('time_groups_visualization.png')
        plt.close()

    def semantic_grouping(self) -> pd.DataFrame:
        """Advanced semantic-based grouping using TF-IDF and cosine similarity."""
        messages = self.messages.copy()
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(messages['text'])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Group messages based on similarity
        current_group = 1
        messages['semantic_group'] = 0
        processed = set()
        
        for i in range(len(messages)):
            if i in processed:
                continue
                
            # Find similar messages
            similar_indices = np.where(similarity_matrix[i] > 0.3)[0]
            
            # Assign group
            messages.loc[messages.index[similar_indices], 'semantic_group'] = current_group
            processed.update(similar_indices)
            current_group += 1
        
        return messages

    def combine_groupings(self) -> pd.DataFrame:
        """Enhanced grouping combination with weighted voting."""
        messages = self.messages.copy()
        
        # Apply individual grouping techniques
        messages = self.time_based_grouping()
        messages = self.semantic_grouping()
        
        # Weighted combination
        weights = {
            'time_group': 0.4,
            'semantic_group': 0.6
        }
        
        # Initialize final groups
        current_group = 1
        messages['final_group'] = 0
        processed = set()
        
        # Combine groups based on weighted similarity
        for i in range(len(messages)):
            if i in processed:
                continue
                
            # Calculate weighted group similarity
            similar_msgs = []
            for j in range(len(messages)):
                if i == j:
                    continue
                    
                similarity = sum(
                    weight * (messages.iloc[i][group] == messages.iloc[j][group])
                    for group, weight in weights.items()
                )
                
                if similarity >= 0.5:
                    similar_msgs.append(j)
            
            # Assign group
            messages.loc[messages.index[similar_msgs + [i]], 'final_group'] = current_group
            processed.update(similar_msgs + [i])
            current_group += 1
        
        return messages

    def generate_report(self, messages: pd.DataFrame) -> None:
        """Generate detailed analysis report."""
        report = {
            'total_messages': len(messages),
            'total_groups': messages['final_group'].nunique(),
            'avg_group_size': len(messages) / messages['final_group'].nunique(),
            'group_sizes': messages['final_group'].value_counts().to_dict(),
            'time_span_hours': (messages['datetime'].max() - messages['datetime'].min()).total_seconds() / 3600
        }
        
        # Save report
        with open('grouping_analysis_report.txt', 'w') as f:
            f.write("Conversation Grouping Analysis Report\n")
            f.write("===================================\n\n")
            for key, value in report.items():
                f.write(f"{key}: {value}\n")

def main():
    """Enhanced main function with better error handling and reporting."""
    parser = argparse.ArgumentParser(description="Advanced pre-grouping for conversation detection")
    parser.add_argument("input_file", help="Input CSV file with messages")
    parser.add_argument("--time-window", type=int, default=30, help="Maximum time window in minutes")
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = MessageAnalyzer(args.input_file)
        
        # Load and process messages
        if analyzer.load_messages() is None:
            return 1
            
        # Apply grouping techniques
        messages = analyzer.combine_groupings()
        
        # Generate report
        analyzer.generate_report(messages)
        
        # Save results
        output_file = os.path.join(os.path.dirname(args.input_file), "enhanced_grouped_messages.csv")
        messages[['id', 'final_group']].to_csv(output_file, index=False)
        logger.info(f"Saved enhanced groupings to {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 