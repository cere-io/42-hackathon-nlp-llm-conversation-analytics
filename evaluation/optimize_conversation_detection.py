#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified Conversation Detection Optimization Workflow
------------------------------------------------------
This script orchestrates a simplified version of the conversation detection workflow.
"""

import os
import argparse
import subprocess
import yaml
import pandas as pd
from datetime import datetime

def run_command(cmd, desc=None):
    """Run a command and return its output."""
    if desc:
        print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def save_config(config, output_path):
    """Save configuration to YAML file."""
    try:
        with open(output_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def run_model_playground(input_file, config_path):
    """Run model playground with specific configuration."""
    # Convert to absolute paths
    abs_input_file = os.path.abspath(input_file)
    abs_config_path = os.path.abspath(config_path)
    
    cmd = ["python", "open_source_examples/model_playground.py", abs_input_file, "--config", abs_config_path]
    
    return run_command(cmd, "Model playground execution")

def compute_metrics(group_dir):
    """Compute metrics on results."""
    abs_group_dir = os.path.abspath(group_dir)
    cmd = ["python", "conversation_metrics.py", abs_group_dir]
    return run_command(cmd, "Metric computation")

def main():
    """Main function to orchestrate the workflow."""
    parser = argparse.ArgumentParser(description="Optimize conversation detection workflow (simplified)")
    parser.add_argument("input_file", help="Path to the input CSV file with messages")
    parser.add_argument("--config", default="open_source_examples/model_config.yaml", 
                        help="Base configuration file")
    parser.add_argument("--prompt", default="open_source_examples/prompts/CD_prompt_simple.txt",
                       help="Optimized prompt to use")
    parser.add_argument("--output-dir", default="results",
                       help="Directory to store optimization results")
    args = parser.parse_args()
    
    # Convert all paths to absolute paths
    input_file = os.path.abspath(args.input_file)
    config_file = os.path.abspath(args.config)
    prompt_file = os.path.abspath(args.prompt)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    config_dir = os.path.join(base_output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    # Save original configuration
    base_config = load_config(config_file)
    if base_config is None:
        return
    
    original_config_path = os.path.join(config_dir, "original_config.yaml")
    save_config(base_config, original_config_path)
    
    # Update config to use deepseek-coder and optimized prompt
    optimized_config = base_config.copy()
    optimized_config['model'] = {'names': ["deepseek-coder:latest"],
                                'temperature': 0.1,
                                'top_p': 0.9,
                                'max_tokens': 1500}
    optimized_config['prompt'] = {'path': prompt_file}
    optimized_config['processing'] = {'batch_size': 6,
                                      'max_context_messages': 10,
                                      'min_confidence_threshold': 0.8}
    
    optimized_config_path = os.path.join(config_dir, "optimized_config.yaml")
    save_config(optimized_config, optimized_config_path)
    
    # Compute initial metrics on existing results
    group_dir = os.path.dirname(input_file)
    initial_metrics = compute_metrics(group_dir)
    
    # Look for metrics file
    metrics_file = None
    if initial_metrics:
        for line in initial_metrics.split('\n'):
            if "Saved metrics to" in line:
                metrics_file = line.split("Saved metrics to")[1].strip()
                break
    
    print("\n===== USING OPTIMIZED CONFIGURATION =====")
    # Run final model with optimized config
    final_output = run_model_playground(input_file, optimized_config_path)
    
    # Compute final metrics
    final_metrics = compute_metrics(group_dir)
    
    # Print results
    print("\n===== OPTIMIZATION COMPLETE =====")
    print(f"All results saved to: {base_output_dir}")
    
    # Read and display the final metrics
    if metrics_file and os.path.exists(metrics_file):
        try:
            df = pd.read_csv(metrics_file)
            print("\n===== FINAL METRICS =====")
            print(df.to_string(index=False))
            
            # Find best model
            best_model = df.loc[df['ari'].idxmax()]
            print("\n===== BEST MODEL =====")
            print(f"Model: {best_model['model']}")
            print(f"File: {best_model['label_file']}")
            print(f"ARI Score: {best_model['ari']:.4f}")
        except Exception as e:
            print(f"Error reading metrics file: {e}")

if __name__ == "__main__":
    main() 