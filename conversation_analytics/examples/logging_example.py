"""
Example demonstrating the secure logging system.
"""

from conversation_analytics.utils.logging_config import setup_logging, get_logger

def main():
    # Setup logging configuration
    setup_logging()
    
    # Get logger instance
    logger = get_logger(__name__)
    
    # Test sensitive data filtering in string messages
    logger.info("Starting application with API key: abc123xyz")
    logger.debug("User token=sensitive_token_123")
    logger.warning("Database password='super_secret'")
    
    # Test sensitive data filtering in dictionaries
    user_data = {
        "username": "john_doe",
        "password": "secret123",
        "api_key": "xyz789abc",
        "settings": {
            "token": "jwt_token_here",
            "preferences": {
                "theme": "dark",
                "secret_key": "nested_secret"
            }
        }
    }
    
    logger.info("User data: %s", user_data)
    
    # Test sensitive data in error context
    try:
        raise ValueError("Failed to authenticate with token=abc123")
    except Exception as e:
        logger.error("Error occurred: %s", str(e), exc_info=True)

    # Test non-sensitive data remains unchanged
    logger.info("Normal message with no secrets")
    logger.debug({
        "user": "john",
        "theme": "dark",
        "status": "active"
    })

if __name__ == "__main__":
    main() 