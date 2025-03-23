# Conversation Analytics System

## Description
Advanced system for detecting and grouping conversations in chat messages using natural language processing and machine learning techniques.

## Key Features
- Automatic conversation detection using multiple techniques
- Pre-grouping based on time, semantics, and user patterns
- Comprehensive evaluation with multiple metrics
- Detailed result visualizations
- Support for different language models

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`
- GPU optional (recommended for large models)

## Installation

### Using pip
```bash
pip install conversation-analytics
```

### From source
```bash
# Clone the repository
git clone https://github.com/zyra-audition/conversation-analytics.git
cd conversation-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r config/requirements.txt

# Install in development mode
pip install -e .
```

## Project Structure
```
.
├── src/
│   ├── processors/     # Data processing components
│   ├── detectors/      # Detection algorithms
│   ├── metrics/        # Evaluation metrics
│   ├── models/         # ML model implementations
│   └── utils/          # Utility functions
├── optimization/       # Optimization scripts
├── evaluation/        # Evaluation scripts
├── tests/            # Unit tests
├── data/             # Input/output data
└── results/          # Results and visualizations
```

## Usage

### Basic Usage
```python
from conversation_analytics import ConversationDetector, TextVectorizer

# Initialize components
vectorizer = TextVectorizer()
detector = ConversationDetector()

# Process messages
messages = [...]  # Your messages here
conversations = detector.detect(messages)
```

### Command Line Interface
```bash
# Pre-group messages
conversation-analytics pre-group data/groups/example/messages.csv

# Evaluate results
conversation-analytics evaluate data/groups/example

# Optimize models
conversation-analytics optimize data/groups/example/messages.csv
```

## API Documentation

### TextVectorizer
```python
class TextVectorizer:
    """
    Converts text data into numerical vectors using TF-IDF.
    
    Args:
        max_features (int): Maximum number of features to extract
        n_components (int, optional): Number of components for dimensionality reduction
        cache_ttl (int): Time-to-live for vector cache in seconds
    """
```

### ConversationDetector
```python
class ConversationDetector:
    """
    Detects and groups conversations in message sequences.
    
    Args:
        batch_size (int): Number of messages to process in each batch
        cache_dir (str): Directory for cache storage
    """
```

## Metrics and Evaluation
The system uses multiple metrics to evaluate grouping quality:
- Adjusted Rand Index (ARI)
- Temporal coherence
- Semantic coherence
- Group statistics

## Visualizations
- ARI score plots over time
- Coherence metrics
- Correlation heatmaps
- Temporal group visualizations

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy .
```

### Security Checks
```bash
# Run security checks
bandit -r src/
safety check
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add type hints to all functions
- Write docstrings for all public APIs
- Add tests for new features
- Update documentation as needed

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- Email: zyra.v23@protonmail.com 
- GitHub: [@zyra-audition](https://github.com/zyra-audition)

## Acknowledgments
- Special thanks to all contributors
- Inspired by best practices in NLP and conversation analysis
- Based on recent research in natural language processing 