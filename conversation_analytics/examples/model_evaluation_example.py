"""
Enhanced example script demonstrating advanced model evaluation capabilities.

This script shows how to:
1. Generate realistic conversation features
2. Evaluate multiple models with cross-validation
3. Compare performance across different metrics
4. Generate detailed visualizations and reports
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time
import psutil
from typing import Dict, Tuple, List
import random
import string

from conversation_analytics.model_evaluation import ModelEvaluator

def generate_realistic_features(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate more realistic conversation features."""
    # Generate random conversations
    conversations = []
    for _ in range(n_samples):
        # Message length (words)
        msg_length = random.randint(5, 100)
        
        # Average word length
        avg_word_length = random.uniform(3.0, 8.0)
        
        # Special characters ratio
        special_chars_ratio = random.uniform(0.0, 0.3)
        
        # Response time (seconds)
        response_time = random.uniform(1.0, 300.0)
        
        # Message frequency (msgs/hour)
        msg_frequency = random.uniform(1.0, 20.0)
        
        # Unique words ratio
        unique_words_ratio = random.uniform(0.3, 1.0)
        
        # Sentiment score (-1 to 1)
        sentiment = random.uniform(-1.0, 1.0)
        
        # Question frequency
        question_freq = random.uniform(0.0, 0.5)
        
        # URL presence
        url_presence = random.randint(0, 1)
        
        conversations.append([
            msg_length, avg_word_length, special_chars_ratio,
            response_time, msg_frequency, unique_words_ratio,
            sentiment, question_freq, url_presence
        ])
    
    X = np.array(conversations)
    
    # Generate labels (spam vs not spam)
    # A message is more likely to be spam if it has:
    # - High special characters ratio
    # - Very short or very long length
    # - Quick response time
    # - High message frequency
    # - URLs present
    spam_scores = (
        X[:, 2] * 2 +  # special chars weight
        np.abs(X[:, 0] - 50) / 50 +  # penalize extreme lengths
        (1 - X[:, 3] / 300) +  # quick responses
        (X[:, 4] / 20) +  # high frequency
        X[:, 8] * 1.5  # URL presence
    ) / 6

    y = (spam_scores > 0.5).astype(int)
    
    return X, y

def evaluate_model_with_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5
) -> Dict:
    """Evaluate model using cross-validation."""
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    start_time = time.time()
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate performance metrics
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    results = {
        'test_scores': {
            metric: cv_results[f'test_{metric}'].mean()
            for metric in scoring.keys()
        },
        'train_scores': {
            metric: cv_results[f'train_{metric}'].mean()
            for metric in scoring.keys()
        },
        'performance_metrics': {
            'processing_time': (end_time - start_time) / cv_folds,
            'memory_usage': end_memory - start_memory,
            'fit_times': cv_results['fit_time'].mean(),
            'score_times': cv_results['score_time'].mean()
        }
    }
    
    return results

def main():
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir="evaluation_results")
    
    # Generate realistic data
    print("Generating realistic conversation data...")
    X, y = generate_realistic_features(n_samples=2000)
    
    # Define models to evaluate
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "svm": SVC(
            probability=True,
            random_state=42
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(50, 25),
            max_iter=1000,
            random_state=42
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        )
    }
    
    # Evaluate each model with cross-validation
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        results = evaluate_model_with_cv(model, X, y)
        
        # Format results for evaluator
        evaluator.evaluate_model(
            model_name=model_name,
            predictions=model.fit(X, y).predict(X),  # For confusion matrix
            ground_truth=y,
            probabilities=model.fit(X, y).predict_proba(X)[:, 1],
            performance_metrics={
                **results['performance_metrics'],
                'cv_accuracy': results['test_scores']['accuracy'],
                'cv_precision': results['test_scores']['precision'],
                'cv_recall': results['test_scores']['recall'],
                'cv_f1': results['test_scores']['f1'],
                'cv_roc_auc': results['test_scores']['roc_auc'],
                'train_accuracy': results['train_scores']['accuracy'],
                'train_precision': results['train_scores']['precision'],
                'train_recall': results['train_scores']['recall'],
                'train_f1': results['train_scores']['f1'],
                'train_roc_auc': results['train_scores']['roc_auc']
            }
        )
    
    # Compare models
    print("\nModel Comparison:")
    comparison_df = evaluator.compare_models(list(models.keys()))
    print(comparison_df)
    
    # Generate plots for each metric
    metrics_to_plot = [
        'cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1', 'cv_roc_auc',
        'processing_time', 'memory_usage'
    ]
    
    for metric in metrics_to_plot:
        if metric in comparison_df.columns:
            print(f"\nGenerating plot for {metric}...")
            evaluator.plot_comparison(metric)
    
    # Generate comprehensive report
    report_path = evaluator.generate_report(include_plots=True)
    print(f"\nReport generated: {report_path}")

if __name__ == "__main__":
    main() 