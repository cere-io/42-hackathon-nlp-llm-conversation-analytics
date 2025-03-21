"""
Tests for the command-line interface.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.cli import main, setup_logging, pre_group_messages, evaluate_results, optimize_models

# Fixtures
@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory with test files."""
    # Create input directory
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # Create sample input file
    input_file = input_dir / "messages.csv"
    input_file.write_text(
        "timestamp,user,text\n"
        "2024-01-01 10:00:00,user1,Hello\n"
        "2024-01-01 10:00:05,user2,Hi there\n"
        "2024-01-01 10:00:10,user1,How are you?\n"
    )
    
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    return tmp_path

# CLI Tests
def test_setup_logging():
    """Test logging setup."""
    with patch('logging.basicConfig') as mock_basic_config:
        setup_logging('DEBUG')
        mock_basic_config.assert_called_once_with(
            level='DEBUG',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

def test_pre_group_messages(temp_dir):
    """Test message pre-grouping command."""
    input_file = temp_dir / "input" / "messages.csv"
    output_dir = temp_dir / "output"
    
    with patch('src.cli.TextVectorizer') as mock_vectorizer, \
         patch('src.cli.ConversationDetector') as mock_detector:
        
        # Configure mocks
        mock_vectorizer.return_value.transform.return_value = MagicMock()
        mock_detector.return_value.detect.return_value = [
            {
                "messages": [{"text": "Hello", "user": "user1"}],
                "start_time": "2024-01-01 10:00:00",
                "end_time": "2024-01-01 10:00:10",
                "participants": {"user1", "user2"}
            }
        ]
        
        # Run pre-grouping
        pre_group_messages(input_file, output_dir)
        
        # Check that output file was created
        output_file = output_dir / "pre_grouped_messages.csv"
        assert output_file.exists()

def test_evaluate_results(temp_dir):
    """Test results evaluation command."""
    with patch('src.cli.evaluate_conversations') as mock_evaluate:
        # Configure mock
        mock_evaluate.return_value = {
            'num_conversations': 2,
            'avg_conversation_length': 3,
            'avg_participants': 2
        }
        
        # Run evaluation
        metrics = evaluate_results(temp_dir / "output")
        
        # Check metrics
        assert metrics['num_conversations'] == 2
        assert metrics['avg_conversation_length'] == 3
        assert metrics['avg_participants'] == 2

def test_optimize_models(temp_dir):
    """Test model optimization command."""
    with patch('src.cli.OptimizationModel') as mock_optimizer:
        # Configure mock
        mock_optimizer.return_value.optimize.return_value = {
            'best_params': {'n_estimators': 100},
            'best_score': 0.95
        }
        
        # Run optimization
        results = optimize_models(temp_dir / "input", temp_dir / "output")
        
        # Check results
        assert 'best_params' in results
        assert 'best_score' in results
        assert results['best_score'] == 0.95

def test_main_help():
    """Test main function help message."""
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        with patch('sys.argv', ['cli.py', '--help']):
            main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 0

def test_main_pre_group():
    """Test main function with pre-group command."""
    with patch('sys.argv', [
        'cli.py',
        'pre-group',
        '--input', 'input.csv',
        '--output', 'output'
    ]), patch('src.cli.pre_group_messages') as mock_pre_group:
        main()
        mock_pre_group.assert_called_once()

def test_main_evaluate():
    """Test main function with evaluate command."""
    with patch('sys.argv', [
        'cli.py',
        'evaluate',
        '--directory', 'output'
    ]), patch('src.cli.evaluate_results') as mock_evaluate:
        main()
        mock_evaluate.assert_called_once()

def test_main_optimize():
    """Test main function with optimize command."""
    with patch('sys.argv', [
        'cli.py',
        'optimize',
        '--input', 'input',
        '--output', 'output'
    ]), patch('src.cli.optimize_models') as mock_optimize:
        main()
        mock_optimize.assert_called_once()

def test_main_log_level():
    """Test main function with log level setting."""
    with patch('sys.argv', [
        'cli.py',
        '--log-level', 'DEBUG',
        'pre-group',
        '--input', 'input.csv',
        '--output', 'output'
    ]), patch('src.cli.setup_logging') as mock_setup_logging:
        main()
        mock_setup_logging.assert_called_with('DEBUG')

# Error Handling Tests
def test_main_invalid_command():
    """Test main function with invalid command."""
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        with patch('sys.argv', ['cli.py', 'invalid']):
            main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code != 0

def test_main_missing_arguments():
    """Test main function with missing arguments."""
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        with patch('sys.argv', ['cli.py', 'pre-group']):
            main()
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code != 0

def test_main_invalid_file():
    """Test main function with invalid file."""
    with pytest.raises(FileNotFoundError):
        with patch('sys.argv', [
            'cli.py',
            'pre-group',
            '--input', 'nonexistent.csv',
            '--output', 'output'
        ]):
            main() 