import pandas as pd
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from advanced_processor import AdvancedProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_messages(file_path: str, sample_size: int = 5):
    """
    Process messages from a CSV file and show processing examples.
    
    Args:
        file_path (str): Path to CSV file containing messages
        sample_size (int): Number of messages to show as examples
    """
    try:
        # Load data
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Initialize processor
        processor = AdvancedProcessor()
        
        # Process messages
        logger.info("Processing messages...")
        processed_messages = []
        all_texts = []
        
        for idx, row in df.iterrows():
            if pd.isna(row['text']):
                continue
                
            result = processor.process_message(row['text'])
            processed_messages.append(result)
            all_texts.append(row['text'])
            
            if idx < sample_size:
                logger.info("\nProcessing example:")
                logger.info(f"Original message: {row['text']}")
                logger.info(f"Clean text: {result['cleaned_text']}")
                logger.info(f"Tokens: {result['tokens']}")
                logger.info(f"Sentiment: {result['sentiment_analysis']['sentiment']} "
                          f"(polarity: {result['sentiment_analysis']['polarity']:.2f}, "
                          f"subjectivity: {result['sentiment_analysis']['subjectivity']:.2f})")
                logger.info(f"Keywords: {[k['keyword'] for k in result['keywords']]}")
                logger.info(f"Is spam: {result['spam_analysis']['is_spam']} "
                          f"(score: {result['spam_analysis']['spam_score']:.2f})")
                logger.info("-" * 50)
        
        # Detect main topics
        logger.info("\nDetecting main topics...")
        topics = processor.detect_topics(all_texts)
        
        for topic in topics:
            logger.info(f"\nTopic {topic['topic_id']}:")
            logger.info(f"Keywords: {topic['keywords']}")
            logger.info(f"Cluster size: {topic['size']} messages")
        
        # Calculate statistics
        total_messages = len(processed_messages)
        spam_messages = sum(1 for msg in processed_messages if msg['spam_analysis']['is_spam'])
        sentiment_counts = {
            'positive': sum(1 for msg in processed_messages 
                          if msg['sentiment_analysis']['sentiment'] == 'positive'),
            'negative': sum(1 for msg in processed_messages 
                          if msg['sentiment_analysis']['sentiment'] == 'negative'),
            'neutral': sum(1 for msg in processed_messages 
                          if msg['sentiment_analysis']['sentiment'] == 'neutral')
        }
        
        # Print summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total messages processed: {total_messages}")
        logger.info(f"Spam messages detected: {spam_messages} ({spam_messages/total_messages*100:.1f}%)")
        logger.info("\nSentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            logger.info(f"- {sentiment.capitalize()}: {count} ({count/total_messages*100:.1f}%)")
            
    except Exception as e:
        logger.error(f"Error processing messages: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    process_messages("data/messages.csv") 