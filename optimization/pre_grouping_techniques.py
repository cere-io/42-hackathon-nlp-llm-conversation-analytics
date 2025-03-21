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
import sys

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
            min_df=1,
            max_df=0.95
        )
        
    def load_messages(self) -> Optional[pd.DataFrame]:
        """Load and preprocess messages from CSV file."""
        try:
            # Read the CSV file and print its columns
            df = pd.read_csv(self.input_file)
            print(f"Loaded {len(df)} messages from {self.input_file}")
            print(f"Available columns: {', '.join(df.columns)}")
            
            # Try to identify key columns by checking all columns
            id_col = None
            text_col = None
            timestamp_col = None
            username_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                # Identify ID column
                if 'id' in col_lower and not any(x in col_lower for x in ['user', 'conv', 'conversation']):
                    id_col = col
                # Identify text column
                elif any(x in col_lower for x in ['text', 'content', 'message']):
                    text_col = col
                # Identify timestamp column
                elif any(x in col_lower for x in ['time', 'date', 'timestamp']):
                    timestamp_col = col
                # Identify username column
                elif any(x in col_lower for x in ['user', 'name', 'author']):
                    username_col = col
            
            # Map identified columns
            column_mapping = {}
            if id_col:
                column_mapping[id_col] = 'id'
            if text_col:
                column_mapping[text_col] = 'text'
            if timestamp_col:
                column_mapping[timestamp_col] = 'timestamp'
            if username_col:
                column_mapping[username_col] = 'username'
                
            # Rename columns if found
            if column_mapping:
                df = df.rename(columns=column_mapping)
                print(f"Mapped columns: {column_mapping}")
            
            # Add missing columns with placeholder values if needed
            required_columns = ['id', 'text', 'timestamp', 'username']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Error: {col.capitalize()} column not found in messages")
                    if col == 'id':
                        df['id'] = range(1, len(df) + 1)
                    elif col == 'text':
                        df['text'] = "No text available"
                    elif col == 'username':
                        df['username'] = "unknown_user"
                    elif col == 'timestamp':
                        df['timestamp'] = datetime.now().isoformat()
            
            # Data cleaning
            df['text'] = df['text'].fillna('').astype(str)
            df['username'] = df['username'].fillna('unknown_user')
            
            # Convert timestamps
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            except:
                print("Warning: Could not parse timestamp column, using current time")
                df['datetime'] = pd.to_datetime(datetime.now())
            
            # Basic statistics
            self._log_data_statistics(df)
            
            self.messages = df
            return df
            
        except Exception as e:
            print(f"Error loading messages: {str(e)}")
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
        
        try:
            # Calculate message density
            time_diffs = messages['datetime'].diff().dt.total_seconds() / 60
            avg_time_diff = time_diffs.mean()
            
            # Adjust window based on message density
            adaptive_window = min(max(avg_time_diff * 2, 5), time_window_minutes)
            print(f"Using adaptive time window of {adaptive_window:.2f} minutes")
            
            current_group = 1
            messages['time_group'] = 0
            
            # Initialize first message with group 1
            if len(messages) > 0:
                messages.iloc[0, messages.columns.get_loc('time_group')] = current_group
                
            # Group messages based on time differences
            for i in range(1, len(messages)):
                time_diff = (messages.iloc[i]['datetime'] - 
                            messages.iloc[i-1]['datetime']).total_seconds() / 60
                
                if time_diff > adaptive_window:
                    current_group += 1
                    
                messages.iloc[i, messages.columns.get_loc('time_group')] = current_group
                
            # Log group statistics
            n_groups = messages['time_group'].nunique()
            avg_group_size = len(messages) / n_groups if n_groups > 0 else 0
            print(f"Created {n_groups} time-based groups with average size {avg_group_size:.2f}")
            
            return messages
        except Exception as e:
            print(f"Error in time-based grouping: {str(e)}")
            print("Warning: time_group not found, adding default values")
            messages['time_group'] = 1
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

    def keyword_based_grouping(self, similarity_threshold: float = 0.3) -> pd.DataFrame:
        """Group messages based on keyword similarity."""
        messages = self.messages.copy()
        
        try:
            # Create document-term matrix
            if len(messages) < 3 or messages['text'].str.len().mean() < 10:
                print("Warning: Not enough text data for keyword analysis")
                messages['keyword_group'] = 1
                return messages
                
            # Create document vectors
            message_texts = messages['text'].fillna('').astype(str).tolist()
            try:
                doc_term_matrix = self.vectorizer.fit_transform(message_texts)
                # Calculate cosine similarity
                similarity_matrix = cosine_similarity(doc_term_matrix)
            except Exception as e:
                print(f"Error in text vectorization: {str(e)}")
                messages['keyword_group'] = 1
                return messages
            
            # Initialize groups
            current_group = 1
            messages['keyword_group'] = 0
            processed = [False] * len(messages)
            
            # Group messages
            for i in range(len(messages)):
                if processed[i]:
                    continue
                    
                messages.iloc[i, messages.columns.get_loc('keyword_group')] = current_group
                processed[i] = True
                
                # Find similar messages
                for j in range(i+1, len(messages)):
                    if not processed[j] and similarity_matrix[i, j] >= similarity_threshold:
                        messages.iloc[j, messages.columns.get_loc('keyword_group')] = current_group
                        processed[j] = True
                        
                current_group += 1
            
            # Log statistics
            n_groups = messages['keyword_group'].nunique()
            avg_group_size = len(messages) / n_groups if n_groups > 0 else 0
            print(f"Created {n_groups} keyword-based groups with average size {avg_group_size:.2f}")
            
            return messages
        except Exception as e:
            print(f"Error in keyword-based grouping: {str(e)}")
            print("Warning: keyword_group not found, adding default values")
            messages['keyword_group'] = 1
            return messages

    def combine_groupings(self) -> pd.DataFrame:
        """Enhanced grouping combination with weighted voting."""
        try:
            messages = self.messages.copy()
            
            # Ensure we have time groups
            if 'time_group' not in messages.columns:
                print("Warning: time_group not found, adding default values")
                messages['time_group'] = 1
                
            # Ensure we have keyword groups    
            if 'keyword_group' not in messages.columns:
                print("Warning: keyword_group not found, adding default values")
                messages['keyword_group'] = 1
            
            # Initialize combined group
            messages['combined_group'] = 1
            
            # If we only have default groups, return early
            if messages['time_group'].nunique() == 1 and messages['keyword_group'].nunique() == 1:
                print("Combined grouping created 1 group")
                return messages
                
            # Otherwise, create similarity matrix based on group memberships
            n_messages = len(messages)
            similarity_matrix = np.zeros((n_messages, n_messages))
            
            # Weights for different grouping methods
            weights = {
                'time_group': 0.6,
                'keyword_group': 0.4
            }
            
            # Build similarity matrix
            for i in range(n_messages):
                for j in range(i+1, n_messages):
                    score = 0.0
                    
                    # Calculate similarity based on group membership
                    for group_type, weight in weights.items():
                        if group_type in messages.columns:
                            if messages.iloc[i][group_type] == messages.iloc[j][group_type]:
                                score += weight
                                
                    similarity_matrix[i, j] = score
                    similarity_matrix[j, i] = score
            
            # Create combined groups
            current_group = 1
            processed = [False] * n_messages
            
            for i in range(n_messages):
                if processed[i]:
                    continue
                    
                # Find similar messages based on weighted score
                similar_indices = []
                for j in range(n_messages):
                    if i != j and not processed[j] and similarity_matrix[i, j] >= 0.4:  # Threshold
                        similar_indices.append(j)
                        
                # Assign to group
                messages.iloc[i, messages.columns.get_loc('combined_group')] = current_group
                processed[i] = True
                
                for j in similar_indices:
                    messages.iloc[j, messages.columns.get_loc('combined_group')] = current_group
                    processed[j] = True
                    
                current_group += 1
                
            print(f"Combined grouping created {messages['combined_group'].nunique()} groups")
            return messages
            
        except Exception as e:
            print(f"Error in combined grouping: {str(e)}")
            # Fallback to simple grouping
            if self.messages is not None:
                self.messages['combined_group'] = 1
                return self.messages
            return None

    def generate_report(self, messages: pd.DataFrame) -> None:
        """Generate detailed analysis report."""
        report = {
            'total_messages': len(messages),
            'total_groups': messages['combined_group'].nunique(),
            'avg_group_size': len(messages) / messages['combined_group'].nunique(),
            'group_sizes': messages['combined_group'].value_counts().to_dict(),
            'time_span_hours': (messages['datetime'].max() - messages['datetime'].min()).total_seconds() / 3600
        }
        
        # Save report
        with open('grouping_analysis_report.txt', 'w') as f:
            f.write("Conversation Grouping Analysis Report\n")
            f.write("===================================\n\n")
            for key, value in report.items():
                f.write(f"{key}: {value}\n")

    def save_output(self, output_path: str = None) -> None:
        """Save grouped messages to CSV file."""
        if self.messages is None:
            print("No messages to save.")
            return
            
        try:
            if output_path is None:
                # Generate output path from input path
                dir_name = os.path.dirname(self.input_file)
                output_path = os.path.join(dir_name, "pre_grouped_messages.csv")
                
            # Prepare output DataFrame
            output_df = self.messages.copy()
            
            # Ensure ID column is preserved
            id_column = None
            for col in output_df.columns:
                if col.lower() in ['id', 'message_id']:
                    id_column = col
                    break
                    
            if id_column is None:
                output_df['id'] = range(1, len(output_df) + 1)
                id_column = 'id'
                
            # Ensure required columns exist
            for col in ['combined_group', 'time_group', 'keyword_group']:
                if col not in output_df.columns:
                    output_df[col] = 1
                    
            # Select columns for output
            columns_to_save = [id_column, 'combined_group', 'time_group', 'keyword_group']
            other_columns = [c for c in output_df.columns if c not in columns_to_save and c != 'datetime']
            columns_to_save.extend(other_columns)
            
            # Save to CSV
            output_df[columns_to_save].to_csv(output_path, index=False)
            print(f"Saved grouped messages to {output_path}")
        except Exception as e:
            print(f"Error writing output: {str(e)}")

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Enhanced message pre-grouping tool")
    parser.add_argument("input_file", help="Input CSV file with messages")
    parser.add_argument("--time-window", type=int, default=30, help="Time window in minutes")
    parser.add_argument("--similarity", type=float, default=0.3, help="Similarity threshold")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()
    
    try:
        # Inicializar y cargar datos
        analyzer = MessageAnalyzer(args.input_file)
        if analyzer.load_messages() is None:
            print("Failed to load messages. Exiting.")
            sys.exit(1)
        
        # Aplicar técnicas de agrupación
        analyzer.messages = analyzer.time_based_grouping(args.time_window)
        
        try:
            analyzer.messages = analyzer.keyword_based_grouping(args.similarity)
        except Exception as e:
            print(f"Error en agrupación por palabras clave, usando valores predeterminados: {str(e)}")
            if 'keyword_group' not in analyzer.messages.columns:
                analyzer.messages['keyword_group'] = 1
        
        # Combinar agrupaciones
        analyzer.messages = analyzer.combine_groupings()
        
        # Guardar resultados
        analyzer.save_output(args.output)
        
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 