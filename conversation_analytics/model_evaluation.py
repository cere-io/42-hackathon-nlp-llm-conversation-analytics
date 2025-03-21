"""
Model Evaluation Module for Conversation Analytics.

This module provides comprehensive tools for evaluating and comparing different
models and configurations in the conversation analytics pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc
)
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from datetime import datetime

class ModelEvaluator:
    """Evaluates and compares model performance across different metrics."""
    
    def __init__(
        self,
        output_dir: str = "evaluation_results",
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            metrics: List of metrics to track. If None, tracks all available metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "processing_time",
            "memory_usage",
            "cv_accuracy",
            "cv_precision",
            "cv_recall",
            "cv_f1",
            "cv_roc_auc"
        ]
        
        self.metrics = metrics or self.default_metrics
        self.results = {}
        self._setup_logging()
        
        # Set up plotting style
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
    
    def _setup_logging(self):
        """Configure logging for the evaluator."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler(
                self.output_dir / "evaluation.log"
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def evaluate_model(
        self,
        model_name: str,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        performance_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate a model's performance across multiple metrics.
        
        Args:
            model_name: Identifier for the model
            predictions: Model predictions
            ground_truth: True labels
            probabilities: Prediction probabilities (for ROC/AUC)
            performance_metrics: Additional metrics (processing time, memory usage)
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {}
        
        # Classification metrics
        conf_matrix = confusion_matrix(ground_truth, predictions)
        class_report = classification_report(
            ground_truth,
            predictions,
            output_dict=True
        )
        
        results["confusion_matrix"] = conf_matrix
        results["classification_report"] = class_report
        
        # ROC/AUC if probabilities provided
        if probabilities is not None:
            fpr, tpr, _ = roc_curve(ground_truth, probabilities)
            results["auc_roc"] = auc(fpr, tpr)
            results["roc_curve"] = {"fpr": fpr, "tpr": tpr}
        
        # Add performance metrics if provided
        if performance_metrics:
            results.update(performance_metrics)
        
        # Store results
        self.results[model_name] = results
        self._save_results(model_name, results)
        
        return results
    
    def compare_models(
        self,
        model_names: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models across specified metrics.
        
        Args:
            model_names: List of model names to compare
            metrics: Specific metrics to compare. If None, uses all common metrics
            
        Returns:
            DataFrame with comparison results
        """
        comparison = {}
        metrics = metrics or self.metrics
        
        for model in model_names:
            if model not in self.results:
                self.logger.warning(f"Model {model} not found in results")
                continue
                
            model_metrics = {}
            results = self.results[model]
            
            # Extract common classification metrics
            if "classification_report" in results:
                report = results["classification_report"]
                model_metrics["accuracy"] = report["accuracy"]
                model_metrics["macro_f1"] = report["macro avg"]["f1-score"]
                model_metrics["weighted_f1"] = report["weighted avg"]["f1-score"]
            
            # Add AUC-ROC if available
            if "auc_roc" in results:
                model_metrics["auc_roc"] = results["auc_roc"]
            
            # Add all available metrics
            for metric in self.metrics:
                if metric in results:
                    model_metrics[metric] = results[metric]
            
            comparison[model] = model_metrics
        
        return pd.DataFrame(comparison).T
    
    def plot_comparison(
        self,
        metric: str,
        model_names: Optional[List[str]] = None,
        plot_type: str = "bar",
        figsize: Tuple[int, int] = (12, 6)
    ):
        """Create enhanced comparison plots."""
        model_names = model_names or list(self.results.keys())
        comparison_df = self.compare_models(model_names)
        
        plt.figure(figsize=figsize)
        
        if plot_type == "bar":
            ax = sns.barplot(
                data=comparison_df.reset_index(),
                x="index",
                y=metric,
                palette="husl"
            )
            
            # Add value labels on top of bars
            for i, v in enumerate(comparison_df[metric]):
                ax.text(
                    i, v, f'{v:.3f}',
                    ha='center',
                    va='bottom'
                )
        else:
            ax = sns.lineplot(
                data=comparison_df.reset_index(),
                x="index",
                y=metric,
                marker='o'
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
        
        # Add a subtle background color
        ax.set_facecolor('#f8f9fa')
        plt.gca().patch.set_facecolor('#f8f9fa')
        plt.gcf().patch.set_facecolor('#ffffff')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"comparison_{metric}_{datetime.now():%Y%m%d_%H%M}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_roc_curves(
        self,
        model_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot ROC curves for multiple models."""
        model_names = model_names or list(self.results.keys())
        
        plt.figure(figsize=figsize)
        
        for model_name in model_names:
            if model_name not in self.results:
                continue
                
            results = self.results[model_name]
            if "roc_curve" not in results:
                continue
                
            fpr = results["roc_curve"]["fpr"]
            tpr = results["roc_curve"]["tpr"]
            auc_score = results["auc_roc"]
            
            plt.plot(
                fpr, tpr,
                label=f'{model_name} (AUC = {auc_score:.3f})',
                linewidth=2
            )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', pad=20, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', frameon=True, facecolor='white')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_path = self.output_dir / f"roc_curves_{datetime.now():%Y%m%d_%H%M}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_confusion_matrices(
        self,
        model_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """Plot confusion matrices for multiple models."""
        model_names = model_names or list(self.results.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(
            1, n_models,
            figsize=(figsize[0], figsize[1]),
            squeeze=False
        )
        axes = axes.ravel()
        
        for idx, (model_name, ax) in enumerate(zip(model_names, axes)):
            if model_name not in self.results:
                continue
                
            results = self.results[model_name]
            if "confusion_matrix" not in results:
                continue
            
            cm = results["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax,
                cbar=False
            )
            
            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        save_path = self.output_dir / f"confusion_matrices_{datetime.now():%Y%m%d_%H%M}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_report(
        self,
        model_names: Optional[List[str]] = None,
        include_plots: bool = True
    ) -> str:
        """Generate an enhanced evaluation report."""
        model_names = model_names or list(self.results.keys())
        report_time = datetime.now().strftime("%Y%m%d_%H%M")
        report_path = self.output_dir / f"evaluation_report_{report_time}.html"
        
        # Generate comparison table
        comparison_df = self.compare_models(model_names)
        
        # Create HTML report with improved styling
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
                <h1>Model Evaluation Report</h1>
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
            
            # Individual Model Details
            for model in model_names:
                f.write(f"""
                <div class="container">
                    <h2>Model: {model}</h2>
                """)
                
                results = self.results[model]
                
                if "classification_report" in results:
                    f.write("""
                    <div class="metric-group">
                        <h3>Classification Report</h3>
                        {}
                    </div>
                    """.format(
                        pd.DataFrame(results["classification_report"]).to_html()
                    ))
                
                f.write("</div>")
            
            if include_plots:
                f.write('<div class="container">')
                f.write('<h2>Visualizations</h2>')
                
                # ROC Curves
                roc_path = self.plot_roc_curves(model_names)
                f.write(f"""
                <div class="plot-container">
                    <h3>ROC Curves</h3>
                    <img src="{roc_path.name}">
                </div>
                """)
                
                # Confusion Matrices
                cm_path = self.plot_confusion_matrices(model_names)
                f.write(f"""
                <div class="plot-container">
                    <h3>Confusion Matrices</h3>
                    <img src="{cm_path.name}">
                </div>
                """)
                
                # Metric Comparisons
                for metric in self.metrics:
                    if metric in comparison_df.columns:
                        plot_path = self.plot_comparison(metric, model_names)
                        f.write(f"""
                        <div class="plot-container">
                            <h3>{metric.replace('_', ' ').title()} Comparison</h3>
                            <img src="{plot_path.name}">
                        </div>
                        """)
                
                f.write('</div>')
            
            f.write("</body></html>")
        
        self.logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _save_results(self, model_name: str, results: Dict):
        """Save evaluation results to disk."""
        save_path = self.output_dir / f"{model_name}_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(save_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved: {save_path}") 