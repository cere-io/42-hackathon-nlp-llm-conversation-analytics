from conversation_analytics.text_processor import TextProcessor
import json

def main():
    # Initialize text processor
    processor = TextProcessor()
    
    # Example texts to analyze
    example_texts = [
        "I love this new product, it's incredibly useful and easy to use!",
        "Customer service was terrible, I wasted a lot of time and they didn't solve anything.",
        "The project is progressing as planned, with no major issues.",
        "The new software update includes significant performance improvements."
    ]
    
    # Process each text and display results
    for i, text in enumerate(example_texts, 1):
        print(f"\nAnalysis of Text #{i}:")
        print("-" * 50)
        print(f"Original text: {text}")
        
        # Process the text
        results = processor.process_text(text)
        
        # Display formatted results
        print("\nAnalysis results:")
        print(f"Sentiment:")
        print(f"  - Polarity: {results['sentiment']['polarity']:.2f}")
        print(f"  - Subjectivity: {results['sentiment']['subjectivity']:.2f}")
        
        print("\nNoun phrases found:")
        for phrase in results['noun_phrases']:
            print(f"  - {phrase}")
            
        print("\nProcessed tokens:")
        print(f"  - Total tokens: {results['token_count']}")
        print(f"  - First 5 tokens: {', '.join(results['tokens'][:5])}")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 