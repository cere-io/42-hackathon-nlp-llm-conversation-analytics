"""
Command-line interface for the conversation analytics system.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from .detectors.conversation_detector import ConversationDetector
from .metrics.conversation_metrics import evaluate_conversations
from .processors.text_vectorizer import TextVectorizer

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def pre_group_messages(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Pre-group messages using time and semantic similarity.
    
    Args:
        input_file: Path to input CSV file
        output_file: Optional path to output CSV file
    """
    logger.info(f"Pre-grouping messages from {input_file}")
    
    # Initialize components
    vectorizer = TextVectorizer()
    detector = ConversationDetector()
    
    # Process messages
    # TODO: Implement message loading and processing
    
    if output_file:
        logger.info(f"Saving results to {output_file}")
        # TODO: Implement result saving

def evaluate_results(data_dir: str) -> None:
    """
    Evaluate conversation detection results.
    
    Args:
        data_dir: Directory containing conversation data
    """
    logger.info(f"Evaluating results in {data_dir}")
    
    # Run evaluation
    metrics = evaluate_conversations(data_dir)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

def optimize_models(input_file: str) -> None:
    """
    Optimize model parameters.
    
    Args:
        input_file: Path to input data file
    """
    logger.info(f"Optimizing models using {input_file}")
    
    # TODO: Implement model optimization
    pass

def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Conversation Analytics CLI")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Pre-group command
    pre_group_parser = subparsers.add_parser("pre-group", help="Pre-group messages")
    pre_group_parser.add_argument("input_file", help="Input CSV file")
    pre_group_parser.add_argument("--output", help="Output CSV file")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate results")
    eval_parser.add_argument("data_dir", help="Data directory")
    
    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize models")
    opt_parser.add_argument("input_file", help="Input data file")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == "pre-group":
        pre_group_messages(args.input_file, args.output)
    elif args.command == "evaluate":
        evaluate_results(args.data_dir)
    elif args.command == "optimize":
        optimize_models(args.input_file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 