"""
Example demonstrating the secure logging system.
"""

from conversation_analytics.utils.logging_config import setup_logging, get_logger

def main():
    # Setup logging configuration
    setup_logging()
    
    # Get logger instance
    logger = get_logger(__name__)
    
    # Simulate a real-world scenario
    logger.info("Starting conversation analysis")
    
    # Example of processing user data
    user_data = {
        "user_id": "user123",
        "email": "user@example.com",
        "api_key": "sk_live_123456789",  # This will be masked
        "preferences": {
            "theme": "dark",
            "language": "es"
        }
    }
    
    logger.info("Processing user data: %s", user_data)
    
    # Example of error handling
    try:
        # Simulate an API call
        raise ConnectionError("Failed to connect to API with token=abc123")
    except Exception as e:
        logger.error("API connection error: %s", str(e), exc_info=True)
        logger.info("Retrying with backup endpoint...")

if __name__ == "__main__":
    main() 