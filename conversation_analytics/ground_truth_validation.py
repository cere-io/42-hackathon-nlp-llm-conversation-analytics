"""
Ground Truth Validation Module.

This module provides functionality to evaluate model predictions against ground truth data,
including conversation clustering and spam detection validation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)

class GroundTruthValidator:
    """Validates model predictions against ground truth data."""
    
    def __init__(
        self,
        ground_truth_path: str,
        output_dir: str = "validation_results"
    ):
        """
        Initialize the validator.
        
        Args:
            ground_truth_path: Path to ground truth CSV file
            output_dir: Directory to save validation results
        """
        self.ground_truth = pd.read_csv(ground_truth_path)
        self.ground_truth.columns = ['message_id', 'conversation_id']  # Rename columns to match predictions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loaded ground truth data with {len(self.ground_truth)} entries")
        
        # Initialize results storage
        self.results = {}
    
    def validate_predictions(
        self,
        predictions_path: str,
        model_name: str
    ) -> Dict:
        """
        Validate model predictions against ground truth.
        
        Args:
            predictions_path: Path to predictions CSV file
            model_name: Name of the model being validated
            
        Returns:
            Dictionary containing validation metrics
        """
        # Load predictions
        predictions = pd.read_csv(predictions_path)
        
        # Merge with ground truth
        merged_data = pd.merge(
            self.ground_truth,
            predictions[['message_id', 'conversation_id']],
            on='message_id',
            suffixes=('_true', '_pred')
        )
        
        if len(merged_data) == 0:
            logger.warning(f"No matching message IDs found for {model_name}")
            return {}
        
        # Calculate clustering metrics
        metrics = {
            'adjusted_rand': adjusted_rand_score(
                merged_data['conversation_id_true'],
                merged_data['conversation_id_pred']
            ),
            'adjusted_mutual_info': adjusted_mutual_info_score(
                merged_data['conversation_id_true'],
                merged_data['conversation_id_pred']
            ),
            'normalized_mutual_info': normalized_mutual_info_score(
                merged_data['conversation_id_true'],
                merged_data['conversation_id_pred']
            ),
            'homogeneity': homogeneity_score(
                merged_data['conversation_id_true'],
                merged_data['conversation_id_pred']
            ),
            'completeness': completeness_score(
                merged_data['conversation_id_true'],
                merged_data['conversation_id_pred']
            ),
            'v_measure': v_measure_score(
                merged_data['conversation_id_true'],
                merged_data['conversation_id_pred']
            ),
            'coverage': len(merged_data) / len(self.ground_truth)
        }
        
        # Add confidence statistics if available
        if 'confidence' in predictions.columns:
            metrics.update({
                'mean_confidence': predictions['confidence'].mean(),
                'min_confidence': predictions['confidence'].min(),
                'max_confidence': predictions['confidence'].max()
            })
        
        # Calculate conversation statistics
        metrics.update({
            'n_conversations_true': len(merged_data['conversation_id_true'].unique()),
            'n_conversations_pred': len(merged_data['conversation_id_pred'].unique()),
            'n_messages': len(merged_data),
            'avg_messages_per_conv_true': len(merged_data) / len(merged_data['conversation_id_true'].unique()),
            'avg_messages_per_conv_pred': len(merged_data) / len(merged_data['conversation_id_pred'].unique())
        })
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': merged_data
        }
        self._save_results(model_name, metrics)
        
        return metrics
    
    def compare_models(
        self,
        model_names: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Compare metrics across different models.
        
        Args:
            model_names: List of model names to compare. If None, uses all available.
            
        Returns:
            DataFrame with model comparisons
        """
        model_names = model_names or list(self.results.keys())
        comparison = {}
        
        for model in model_names:
            if model not in self.results:
                logger.warning(f"Model {model} not found in results")
                continue
            comparison[model] = self.results[model]['metrics']
        
        return pd.DataFrame(comparison).T
    
    def plot_comparison(
        self,
        metric: str,
        model_names: Optional[list] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Create comparison plot for a specific metric.
        
        Args:
            metric: Metric to plot
            model_names: Models to include. If None, uses all models.
            figsize: Figure size (width, height)
        """
        comparison_df = self.compare_models(model_names)
        
        if metric not in comparison_df.columns:
            logger.error(f"Metric {metric} not found in results")
            return
        
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            data=comparison_df.reset_index(),
            x='index',
            y=metric,
            palette='husl'
        )
        
        # Add value labels
        for i, v in enumerate(comparison_df[metric]):
            ax.text(
                i, v, f'{v:.3f}',
                ha='center',
                va='bottom'
            )
        
        plt.title(
            f"Model Comparison - {metric}",
            pad=20,
            fontsize=14,
            fontweight='bold'
        )
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        save_path = self.output_dir / f"comparison_{metric}_{datetime.now():%Y%m%d_%H%M}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(
        self,
        model_names: Optional[list] = None,
        include_plots: bool = True
    ) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            model_names: Models to include in report
            include_plots: Whether to include visualization plots
            
        Returns:
            Path to generated report
        """
        report_time = datetime.now().strftime("%Y%m%d_%H%M")
        report_path = self.output_dir / f"validation_report_{report_time}.html"
        
        # Generate comparison table
        comparison_df = self.compare_models(model_names)
        
        # Create HTML report
        with open(report_path, "w") as f:
            f.write("""
            <html>
            <head>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 40px;
                        background-color: #f8f9fa;
                    }
                    .container {
                        background-color: white;
                        padding: 20px;
                        border-radius: 5px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                    }
                    h1, h2, h3 {
                        color: #2c3e50;
                    }
                    table {
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }
                    th, td {
                        padding: 12px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }
                    th {
                        background-color: #f8f9fa;
                    }
                    tr:hover {
                        background-color: #f5f5f5;
                    }
                    .metric-group {
                        margin-bottom: 30px;
                    }
                    .plot-container {
                        text-align: center;
                        margin: 20px 0;
                    }
                    img {
                        max-width: 100%;
                        height: auto;
                        border-radius: 5px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }
                </style>
            </head>
            <body>
            """)
            
            # Header
            f.write("""
            <div class="container">
                <h1>Ground Truth Validation Report</h1>
                <p>Generated on: {}</p>
            </div>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            
            # Model Comparison
            f.write("""
            <div class="container">
                <h2>Model Comparison</h2>
                {}
            </div>
            """.format(comparison_df.to_html(classes='dataframe')))
            
            if include_plots:
                f.write('<div class="container">')
                f.write('<h2>Metric Comparisons</h2>')
                
                # Generate and include plots for each metric
                metrics_to_plot = [
                    'adjusted_rand',
                    'adjusted_mutual_info',
                    'normalized_mutual_info',
                    'homogeneity',
                    'completeness',
                    'v_measure'
                ]
                
                for metric in metrics_to_plot:
                    if metric in comparison_df.columns:
                        self.plot_comparison(metric, model_names)
                        f.write(f"""
                        <div class="plot-container">
                            <h3>{metric.replace('_', ' ').title()}</h3>
                            <img src="comparison_{metric}_{report_time}.png">
                        </div>
                        """)
                
                f.write('</div>')
            
            f.write("</body></html>")
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _save_results(self, model_name: str, results: Dict):
        """Save validation results to disk."""
        save_path = self.output_dir / f"{model_name}_validation.json"
        
        import json
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved: {save_path}") 