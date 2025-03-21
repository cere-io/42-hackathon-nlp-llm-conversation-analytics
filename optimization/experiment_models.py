#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment Manager for Conversation Detection Models
---------------------------------------------------
This script automates the process of testing different models, prompts,
and parameter configurations for conversation detection tasks.
"""

import os
import yaml
import json
import argparse
import subprocess
import pandas as pd
from datetime import datetime
from itertools import product

def load_config(config_path="open_source_examples/model_config.yaml"):
    """Load the configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def update_config(config, updates, output_path=None):
    """Update configuration with new parameters and save to a temporary file."""
    if output_path is None:
        # Use absolute path for temp config
        output_path = os.path.abspath("temp_config.yaml")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Deep copy the config to avoid modifying the original
    updated_config = config.copy()
    
    # Apply updates
    for section, params in updates.items():
        if section in updated_config:
            for key, value in params.items():
                updated_config[section][key] = value
    
    # Write updated config to file
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(updated_config, file, default_flow_style=False, allow_unicode=True)
        return output_path
    except Exception as e:
        print(f"Error writing config file: {e}")
        return None

def run_experiment(input_file, config_path, output_dir=None):
    """Run the model playground with the given configuration."""
    # Use absolute path for config
    config_path = os.path.abspath(config_path)
    cmd = ["python", "open_source_examples/model_playground.py", input_file, "--config", config_path]
    
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            print(f"Error running experiment (return code {result.returncode})")
            return None
        
        # Extract the output file path from the output
        for line in result.stdout.split('\n'):
            if "Results written to:" in line:
                output_file = line.split("Results written to:")[1].strip()
                return output_file
        
        return None
    except Exception as e:
        print(f"Exception running experiment: {e}")
        return None

def compute_metrics(group_dir):
    """Run the conversation metrics script on the results."""
    cmd = ["python", "conversation_metrics.py", group_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error computing metrics: {result.stderr}")
        return None
    
    # Find the metrics file
    for line in result.stdout.split('\n'):
        if "Saved metrics to" in line:
            metrics_file = line.split("Saved metrics to")[1].strip()
            return metrics_file
    
    return None

def load_metrics(metrics_file):
    """Load metrics from the CSV file."""
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return None
    
    return pd.read_csv(metrics_file)

def generate_parameter_combinations(parameter_space):
    """Generate all combinations of parameters to test."""
    sections = []
    values = []
    
    for section, params in parameter_space.items():
        for param, values_list in params.items():
            sections.append((section, param))
            values.append(values_list)
    
    combinations = []
    for combo in product(*values):
        config_update = {}
        for i, (section, param) in enumerate(sections):
            if section not in config_update:
                config_update[section] = {}
            config_update[section][param] = combo[i]
        combinations.append(config_update)
    
    return combinations

def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description="Run experiments with different model configurations")
    parser.add_argument("input_file", help="Path to the input CSV file with messages")
    parser.add_argument("--config", default="open_source_examples/model_config.yaml", 
                        help="Base configuration file")
    parser.add_argument("--output", default="experiments", 
                        help="Directory to store experiment results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load base configuration
    base_config = load_config(args.config)
    if base_config is None:
        print("Failed to load base configuration")
        return
    
    # Define parameter space to explore (reduced for testing)
    parameter_space = {
        "model": {
            "temperature": [0.1],
            "top_p": [0.85]
        },
        "processing": {
            "batch_size": [4],
            "max_context_messages": [8]
        },
        "prompt": {
            "path": [
                "open_source_examples/prompts/conversation_detection_prompt.txt"
            ]
        }
    }
    
    # Generate all combinations to test
    combinations = generate_parameter_combinations(parameter_space)
    print(f"Generated {len(combinations)} parameter combinations to test")
    
    # Track results
    results = []
    
    # Run experiments
    for i, config_update in enumerate(combinations):
        print(f"\nExperiment {i+1}/{len(combinations)}")
        print(f"Configuration: {json.dumps(config_update, indent=2)}")
        
        # Update configuration
        temp_config_path = update_config(base_config, config_update)
        if temp_config_path is None:
            print("Failed to update configuration")
            continue
        
        # Run experiment
        output_file = run_experiment(args.input_file, temp_config_path)
        if not output_file:
            print("Experiment failed, skipping")
            continue
        
        # Compute metrics
        group_dir = os.path.dirname(args.input_file)
        metrics_file = compute_metrics(group_dir)
        if not metrics_file:
            print("Failed to compute metrics, skipping")
            continue
        
        # Load and store results
        metrics = load_metrics(metrics_file)
        if metrics is not None:
            # Find the latest result (should be our experiment)
            latest_result = metrics.iloc[-1].to_dict()
            latest_result.update({
                "experiment_id": i+1,
                "config": config_update
            })
            results.append(latest_result)
            print(f"ARI Score: {latest_result.get('ari', 'N/A')}")
    
    # Save all results to CSV
    if results:
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.output, f"experiment_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        
        # Show best result
        best_result = max(results, key=lambda x: x.get('ari', 0))
        print("\n===== BEST CONFIGURATION =====")
        print(f"Experiment ID: {best_result['experiment_id']}")
        print(f"ARI Score: {best_result.get('ari', 'N/A')}")
        print(f"Configuration: {json.dumps(best_result['config'], indent=2)}")
        print(f"Model File: {best_result.get('label_file', 'N/A')}")
        print(f"\nAll results saved to: {results_path}")
    else:
        print("\nNo successful experiments to report")

if __name__ == "__main__":
    main() 