#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prompt Optimization for Conversation Detection
---------------------------------------------
This script enhances prompts based on performance feedback
to improve conversation detection accuracy.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def load_prompt_template(template_path):
    """Load a prompt template from a file."""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        print(f"Loaded prompt template from {template_path}")
        return template
    except Exception as e:
        print(f"Error loading prompt template: {e}")
        return None

def load_metrics(metrics_file):
    """Load metrics from a CSV file."""
    try:
        metrics = pd.read_csv(metrics_file)
        print(f"Loaded metrics from {metrics_file}")
        return metrics
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None

def extract_prompt_id(label_file):
    """Extract prompt ID from label file name."""
    # Example: labels_20250131_143535_gpt4o_thisiscere.csv
    match = re.search(r'(\d+)_(\w+)_', label_file)
    if match:
        return match.group(1), match.group(2)
    return None, None

def analyze_performance(metrics_df, result_files_dir):
    """
    Analyze model performance to identify strengths and weaknesses.
    
    This function examines the results of different models and prompts
    to identify patterns in errors and success cases.
    """
    # Sort by performance
    metrics_df = metrics_df.sort_values('ari', ascending=False)
    
    # Create a dictionary to store performance insights
    insights = {
        "best_model": metrics_df.iloc[0]['model'],
        "best_file": metrics_df.iloc[0]['label_file'],
        "best_ari": metrics_df.iloc[0]['ari'],
        "worst_model": metrics_df.iloc[-1]['model'],
        "worst_file": metrics_df.iloc[-1]['label_file'],
        "worst_ari": metrics_df.iloc[-1]['ari'],
        "avg_ari": metrics_df['ari'].mean(),
        "error_patterns": []
    }
    
    # Load results from best and worst performing models for comparison
    best_file = os.path.join(result_files_dir, insights["best_file"])
    worst_file = os.path.join(result_files_dir, insights["worst_file"])
    
    try:
        best_results = pd.read_csv(best_file)
        worst_results = pd.read_csv(worst_file)
        
        # Find where they differ in conversation assignments
        # This requires a common set of message_ids
        common_ids = set(best_results['message_id']) & set(worst_results['message_id'])
        
        # Filter to common IDs
        best_common = best_results[best_results['message_id'].isin(common_ids)]
        worst_common = worst_results[worst_results['message_id'].isin(common_ids)]
        
        # Ensure same order
        best_common = best_common.sort_values('message_id')
        worst_common = worst_common.sort_values('message_id')
        
        # Find differences
        diff_mask = best_common['conversation_id'] != worst_common['conversation_id']
        best_diff = best_common[diff_mask]
        worst_diff = worst_common[diff_mask]
        
        # Analyze differences
        if len(best_diff) > 0:
            insights["error_patterns"].append({
                "description": "Conversation assignment differences",
                "count": len(best_diff),
                "examples": [
                    {
                        "message_id": best_diff.iloc[i]['message_id'],
                        "best_convo": best_diff.iloc[i]['conversation_id'],
                        "best_topic": best_diff.iloc[i]['topic'],
                        "worst_convo": worst_diff.iloc[i]['conversation_id'],
                        "worst_topic": worst_diff.iloc[i]['topic'],
                    }
                    for i in range(min(3, len(best_diff)))
                ]
            })
    except Exception as e:
        print(f"Error analyzing performance differences: {e}")
    
    return insights

def enhance_prompt(template, insights, enhanced_path):
    """
    Enhance a prompt template based on performance insights.
    
    This function modifies a prompt template to address identified issues
    and improve performance.
    """
    # Create a copy of the template for enhancement
    enhanced = template
    
    # Extract core components
    parts = re.split(r'\[MESSAGES\]', enhanced)
    if len(parts) != 2:
        print("Error: Could not locate [MESSAGES] placeholder in template")
        return None
    
    prefix, suffix = parts
    
    # Enhance instructions based on insights
    enhancements = []
    
    # Add specific improvements based on error patterns
    if insights["error_patterns"]:
        # Example improvement for conversation assignment differences
        convo_diff = next((p for p in insights["error_patterns"] 
                          if p["description"] == "Conversation assignment differences"), None)
        if convo_diff:
            # Add specific examples from error patterns
            examples = convo_diff["examples"]
            if examples:
                example_text = "PAY SPECIAL ATTENTION to these challenging cases:\n"
                for i, ex in enumerate(examples):
                    example_text += f"- Message {ex['message_id']}: Should be grouped by topic '{ex['best_topic']}', not '{ex['worst_topic']}'\n"
                enhancements.append(example_text)
    
    # Add specific model-focused enhancements
    model_enhancements = [
        # General improvements
        "CONTEXTUAL UNDERSTANDING: Consider the entire context of each message, not just keywords.",
        "TEMPORAL CONSISTENCY: Messages within the same time period (30 minutes) with semantic similarity likely belong to the same conversation.",
        "SEMANTIC PRECISION: Focus on the detailed technical meaning of messages, not just surface-level similarities.",
        
        # Specific improvements for low-performing models
        f"AVOID COMMON ERRORS: When assigning conversation IDs, ensure you're grouping by actual topic, not just by user or timestamp."
    ]
    enhancements.extend(model_enhancements)
    
    # Insert enhancements before the [MESSAGES] placeholder
    enhanced_prefix = prefix
    if enhancements:
        # Find a good insertion point
        if "Rules for the analysis" in prefix:
            # Insert before rules
            insertion_point = prefix.find("Rules for the analysis")
            enhanced_prefix = prefix[:insertion_point] + "ENHANCED GUIDANCE:\n" + "\n".join(enhancements) + "\n\n" + prefix[insertion_point:]
        else:
            # Append to the end of the prefix
            enhanced_prefix = prefix + "\nENHANCED GUIDANCE:\n" + "\n".join(enhancements) + "\n\n"
    
    # Reassemble the prompt
    enhanced = enhanced_prefix + "[MESSAGES]" + suffix
    
    # Write the enhanced prompt to file
    try:
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            f.write(enhanced)
        print(f"Enhanced prompt written to {enhanced_path}")
        return enhanced
    except Exception as e:
        print(f"Error writing enhanced prompt: {e}")
        return None

def main():
    """Main function to run prompt optimization."""
    parser = argparse.ArgumentParser(description="Optimize prompts based on performance feedback")
    parser.add_argument("template", help="Path to the prompt template file")
    parser.add_argument("metrics", help="Path to the metrics CSV file")
    parser.add_argument("--output", help="Path to write the enhanced prompt")
    parser.add_argument("--results-dir", default="data/groups/thisiscere",
                        help="Directory containing the result files")
    args = parser.parse_args()
    
    # Set default output if not specified
    if not args.output:
        base_dir = os.path.dirname(args.template)
        base_name = os.path.splitext(os.path.basename(args.template))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(base_dir, f"{base_name}_enhanced_{timestamp}.txt")
    
    # Load prompt template
    template = load_prompt_template(args.template)
    if template is None:
        return
    
    # Load metrics
    metrics = load_metrics(args.metrics)
    if metrics is None:
        return
    
    # Analyze performance
    insights = analyze_performance(metrics, args.results_dir)
    print(f"Performance insights:")
    print(f"- Best model: {insights['best_model']} (ARI: {insights['best_ari']:.4f})")
    print(f"- Worst model: {insights['worst_model']} (ARI: {insights['worst_ari']:.4f})")
    print(f"- Average ARI: {insights['avg_ari']:.4f}")
    print(f"- Found {len(insights['error_patterns'])} error patterns")
    
    # Enhance prompt
    enhanced = enhance_prompt(template, insights, args.output)
    if enhanced is not None:
        print(f"Successfully enhanced prompt template!")

if __name__ == "__main__":
    main() 