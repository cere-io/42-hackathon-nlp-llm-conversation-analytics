#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pre-grouping Techniques for Conversation Detection
-------------------------------------------------
This script implements various pre-grouping techniques to improve
conversation detection performance by providing initial groupings
based on temporal and semantic patterns.
"""

import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from collections import defaultdict

def load_messages(input_file):
    """Load messages from CSV file."""
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} messages from {input_file}")
        return df
    except Exception as e:
        print(f"Error loading messages: {e}")
        return None

def time_based_grouping(messages, time_window_minutes=30):
    """
    Group messages based on temporal proximity.
    
    This technique groups messages that occur within a specified 
    time window, assuming they might be part of the same conversation.
    """
    if 'Timestamp' not in messages.columns:
        print("Error: Timestamp column not found in messages")
        return messages
    
    # Convert timestamps to datetime objects
    try:
        messages['datetime'] = pd.to_datetime(messages['Timestamp'])
    except:
        print("Error: Could not parse timestamps. Ensure format is consistent.")
        return messages
    
    # Sort by timestamp
    messages = messages.sort_values('datetime')
    
    # Initial group assignment
    current_group = 1
    current_time = messages.iloc[0]['datetime']
    groups = []
    
    for _, message in messages.iterrows():
        message_time = message['datetime']
        # If time difference is greater than the window, start a new group
        if (message_time - current_time).total_seconds() / 60 > time_window_minutes:
            current_group += 1
            current_time = message_time
        groups.append(current_group)
    
    # Add group information
    messages['time_group'] = groups
    
    print(f"Time-based grouping created {current_group} groups")
    return messages

def user_interaction_grouping(messages):
    """
    Group messages based on user interaction patterns.
    
    This technique identifies conversation threads by tracking
    which users interact with each other.
    """
    if 'Username' not in messages.columns:
        print("Error: Username column not found in messages")
        return messages
    
    # Initialize interaction graph
    user_interactions = defaultdict(set)
    
    # Build interaction graph based on temporal proximity
    sorted_msgs = messages.sort_values('datetime')
    recent_users = []
    
    for i, row in sorted_msgs.iterrows():
        user = row['Username']
        # Consider interactions with users from recent messages
        for recent_user in recent_users:
            if user != recent_user:
                user_interactions[user].add(recent_user)
                user_interactions[recent_user].add(user)
        
        # Update recent users (keep last 3)
        recent_users.append(user)
        if len(recent_users) > 3:
            recent_users.pop(0)
    
    # Group assignment based on connected components in the interaction graph
    user_group = {}
    current_group = 1
    
    # Assign groups using a simple connected components algorithm
    unvisited = set(user_interactions.keys())
    while unvisited:
        # Start with an unvisited user
        start_user = unvisited.pop()
        user_group[start_user] = current_group
        
        # BFS to find connected users
        queue = [start_user]
        while queue:
            current_user = queue.pop(0)
            for connected_user in user_interactions[current_user]:
                if connected_user in unvisited:
                    unvisited.remove(connected_user)
                    user_group[connected_user] = current_group
                    queue.append(connected_user)
        
        current_group += 1
    
    # Assign groups to messages
    interaction_groups = []
    for _, message in messages.iterrows():
        user = message['Username']
        # If user not in any group, assign to a new group
        if user not in user_group:
            user_group[user] = current_group
            current_group += 1
        interaction_groups.append(user_group[user])
    
    messages['interaction_group'] = interaction_groups
    
    print(f"User interaction grouping created {current_group - 1} groups")
    return messages

def keyword_based_grouping(messages, top_keywords=10):
    """
    Group messages based on common keywords.
    
    This technique identifies conversation threads by tracking
    which messages share important keywords.
    """
    if 'Text' not in messages.columns:
        print("Error: Text column not found in messages")
        return messages
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        min_df=2,
        max_df=0.8
    )
    
    # Fit vectorizer and transform text
    try:
        tfidf_matrix = vectorizer.fit_transform(messages['Text'].fillna(''))
    except Exception as e:
        print(f"Error in TF-IDF vectorization: {e}")
        messages['keyword_group'] = 1  # Default to single group
        return messages
    
    # Get feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()
    
    # For each message, find the most important keywords
    message_keywords = []
    for i in range(tfidf_matrix.shape[0]):
        # Get indices of top keywords for this message
        feature_index = tfidf_matrix[i, :].nonzero()[1]
        # Get scores for those keywords
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        # Sort by score and take top keywords
        top_features = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_keywords]
        # Get keyword indices
        message_keywords.append([feature_names[idx] for idx, score in top_features])
    
    # Build keyword graph (connect messages that share keywords)
    keyword_connections = defaultdict(set)
    
    for i, keywords_i in enumerate(message_keywords):
        for j in range(i+1, len(message_keywords)):
            keywords_j = message_keywords[j]
            # Calculate intersection
            common_keywords = set(keywords_i) & set(keywords_j)
            if len(common_keywords) >= 2:  # At least 2 keywords in common
                keyword_connections[i].add(j)
                keyword_connections[j].add(i)
    
    # Group assignment using connected components
    message_group = {}
    current_group = 1
    
    # Assign groups using a simple connected components algorithm
    unvisited = set(range(len(messages)))
    while unvisited:
        # Start with an unvisited message
        start_msg = unvisited.pop()
        message_group[start_msg] = current_group
        
        # BFS to find connected messages
        queue = [start_msg]
        while queue:
            current_msg = queue.pop(0)
            for connected_msg in keyword_connections[current_msg]:
                if connected_msg in unvisited:
                    unvisited.remove(connected_msg)
                    message_group[connected_msg] = current_group
                    queue.append(connected_msg)
        
        current_group += 1
    
    # Assign groups to messages
    keyword_groups = []
    for i in range(len(messages)):
        # If message not in any group, assign to a new group
        if i not in message_group:
            message_group[i] = current_group
            current_group += 1
        keyword_groups.append(message_group[i])
    
    messages['keyword_group'] = keyword_groups
    
    print(f"Keyword-based grouping created {current_group - 1} groups")
    return messages

def combine_groupings(messages):
    """
    Combine different grouping techniques to create a final grouping.
    
    This creates a consensus grouping that leverages the strengths
    of each individual technique.
    """
    # Ensure all grouping columns exist
    required_columns = ['time_group', 'interaction_group', 'keyword_group']
    for col in required_columns:
        if col not in messages.columns:
            print(f"Warning: {col} not found, adding default values")
            messages[col] = 1  # Default to single group
    
    # Create a new combined grouping
    combined_groups = []
    group_mapping = {}
    
    for i, row in messages.iterrows():
        # Create a tuple of all grouping columns
        group_tuple = (row['time_group'], row['interaction_group'], row['keyword_group'])
        
        # If we haven't seen this combination before, assign a new group
        if group_tuple not in group_mapping:
            group_mapping[group_tuple] = len(group_mapping) + 1
        
        combined_groups.append(group_mapping[group_tuple])
    
    messages['combined_group'] = combined_groups
    
    print(f"Combined grouping created {len(group_mapping)} groups")
    return messages

def output_groupings(messages, output_file):
    """Write pre-grouping results to a CSV file."""
    try:
        # Create a dataframe with required columns
        output_df = messages[['ID', 'combined_group']].copy()
        output_df.columns = ['message_id', 'pre_group']
        
        # Write to CSV
        output_df.to_csv(output_file, index=False)
        print(f"Pre-grouping results written to {output_file}")
        return True
    except Exception as e:
        print(f"Error writing output: {e}")
        return False

def main():
    """Main function to run pre-grouping analysis."""
    parser = argparse.ArgumentParser(description="Apply pre-grouping techniques to conversation data")
    parser.add_argument("input_file", help="Path to the input CSV file with messages")
    parser.add_argument("--output", help="Path to output CSV file")
    parser.add_argument("--time-window", type=int, default=30, 
                        help="Time window in minutes for time-based grouping")
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output:
        base_dir = os.path.dirname(args.input_file)
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = os.path.join(base_dir, f"pre_grouped_{base_name}.csv")
    
    # Load messages
    messages = load_messages(args.input_file)
    if messages is None:
        return
    
    # Apply grouping techniques
    messages = time_based_grouping(messages, args.time_window)
    messages = user_interaction_grouping(messages)
    messages = keyword_based_grouping(messages)
    
    # Combine groupings
    messages = combine_groupings(messages)
    
    # Output results
    output_groupings(messages, args.output)

if __name__ == "__main__":
    main() 