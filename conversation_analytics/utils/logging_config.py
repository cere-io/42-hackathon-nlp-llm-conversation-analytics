"""
Logging configuration module.
"""

import os
import yaml
import logging.config
import tempfile
from typing import Optional
from pathlib import Path

def setup_logging(config_path: Optional[str] = None) -> None:
    """Configure logging using YAML configuration file.
    
    Args:
        config_path: Path to the logging configuration YAML file.
                     If None, uses default config path.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config',
            'logging_config.yaml'
        )
    
    # Use system's temp directory for logs
    log_dir = Path(tempfile.gettempdir()) / 'conversation_analytics' / 'logs'
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create log files if they don't exist
    app_log = log_dir / 'app.log'
    error_log = log_dir / 'error.log'
    app_log.touch(exist_ok=True)
    error_log.touch(exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Update log file paths to use temp directory
    config['handlers']['file']['filename'] = str(app_log)
    config['handlers']['error_file']['filename'] = str(error_log)
    
    logging.config.dictConfig(config)
    
    # Log the location of log files
    logger = logging.getLogger(__name__)
    logger.info(f"Log files are stored in: {log_dir}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger to get
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name) 