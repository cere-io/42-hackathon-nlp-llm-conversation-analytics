"""
Example script demonstrating the usage of GroundTruthValidator.

This script shows how to:
1. Initialize the validator
2. Validate predictions from different models
3. Compare model performances
4. Generate validation reports
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from conversation_analytics.ground_truth_validation import GroundTruthValidator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Define paths
    data_dir = Path("data/groups/thisiscere")
    ground_truth_path = data_dir / "GT_conversations_thisiscere.csv"
    output_dir = "validation_results"
    
    # Initialize validator
    validator = GroundTruthValidator(
        ground_truth_path=str(ground_truth_path),
        output_dir=output_dir
    )
    
    # List of models and their prediction files
    model_predictions = {
        "mistral7b": "labels_20250225_202230_mistral7b_thisiscere.csv",
        "deepseekv3": "labels_20250131_185300_deepseekv3_thisiscere.csv",
        "gpt4o": "labels_20250131_143535_gpt4o_thisiscere.csv",
        "claude35s": "labels_20250131_171944_claude35s_thisiscere.csv"
    }
    
    # Validate each model's predictions
    for model_name, prediction_file in model_predictions.items():
        predictions_path = data_dir / prediction_file
        
        if not predictions_path.exists():
            logger.warning(f"Predictions file not found for {model_name}")
            continue
        
        logger.info(f"Validating predictions for {model_name}")
        metrics = validator.validate_predictions(
            str(predictions_path),
            model_name
        )
        
        logger.info(f"Validation metrics for {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Generate comparison report
    logger.info("Generating validation report...")
    report_path = validator.generate_report(
        model_names=list(model_predictions.keys()),
        include_plots=True
    )
    logger.info(f"Report generated at: {report_path}")
    
    # Compare specific metrics across models
    comparison_df = validator.compare_models(list(model_predictions.keys()))
    logger.info("\nModel Comparison Summary:")
    logger.info("\n" + str(comparison_df))
    
    # Plot comparisons for key metrics
    key_metrics = [
        'adjusted_rand',
        'normalized_mutual_info',
        'v_measure'
    ]
    
    for metric in key_metrics:
        logger.info(f"Generating comparison plot for {metric}")
        validator.plot_comparison(metric, list(model_predictions.keys()))

if __name__ == "__main__":
    main() 