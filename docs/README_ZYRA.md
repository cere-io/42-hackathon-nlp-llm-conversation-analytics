# Zyra Project Guide

## 🚀 Project Overview
This guide provides detailed information about the Zyra project structure, workflow, and development process. It's designed to help developers understand and work with the conversation analytics system effectively.

## 📂 Project Structure
```
zyra-audition/
├── src/                    # Source code
│   ├── detectors/         # Detection modules (spam, conversations)
│   ├── processors/        # Data processing pipelines
│   ├── metrics/          # Evaluation and analysis tools
│   ├── models/           # LLM model integrations
│   └── utils/            # Shared utilities
├── tests/                 # Test suite
│   ├── models/          # Model tests (base, clustering, optimization)
│   ├── detectors/       # Detector tests (spam, conversation)
│   ├── processors/      # Processor tests (text, parallel)
│   ├── metrics/         # Metrics tests (clustering, conversation)
│   └── cli/            # CLI interface tests
├── evaluation/           # Evaluation scripts and tools
├── optimization/         # Performance optimization tools
├── docs/                 # Documentation
├── config/              # Configuration files
├── storage/             # Data storage
│   ├── cache/          # Model and processing caches
│   └── results/        # Analysis results
├── logs/                # Application logs
├── data/                # Input data
└── open_source_examples/ # Example implementations
```

## 🔄 Development Workflow

### 1. Setting Up the Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r config/requirements.txt

# Download NLTK data
python config/download_nltk_data.py
```

### 2. Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/models/     # Run all model tests
python -m pytest tests/detectors/  # Run all detector tests
python -m pytest tests/processors/ # Run all processor tests
python -m pytest tests/metrics/    # Run all metric tests
python -m pytest tests/cli/        # Run all CLI tests

# Run with coverage report
python -m pytest --cov=src tests/
```

### 3. Development Process
1. **Feature Development**
   - Create new branch for feature
   - Implement changes in appropriate directory
   - Add/update tests
   - Update documentation

2. **Testing**
   - Run unit tests
   - Run integration tests
   - Validate performance
   - Check cross-platform compatibility

3. **Code Review**
   - Follow PR template guidelines
   - Update documentation
   - Ensure test coverage
   - Verify performance metrics

## 📊 Reporting and Documentation

### 1. Pull Request Process
- Use the PR template in `docs/PULL_REQUEST_TEMPLATE.md`
- Include relevant metrics and test results
- Update documentation as needed
- Link related issues

### 2. Documentation Updates
- Keep README files up to date
- Document new features
- Update configuration guides
- Maintain API documentation

### 3. Performance Reporting
- Include benchmark results
- Document optimization efforts
- Track model performance
- Monitor resource usage

## 🛠️ Tools and Utilities

### 1. Data Processing
```python
from src.processors.data_processor import DataProcessor
from src.processors.text_vectorizer import TextVectorizer

# Initialize processors
data_processor = DataProcessor()
text_vectorizer = TextVectorizer()

# Process data
processed_data = data_processor.process(raw_data)
vectorized_data = text_vectorizer.vectorize(processed_data)
```

### 2. Model Integration
```python
from src.models.gpt4 import GPT4Detector
from src.models.claude35 import ClaudeDetector

# Initialize detectors
gpt4_detector = GPT4Detector()
claude_detector = ClaudeDetector()

# Run detection
results = gpt4_detector.detect(text)
```

### 3. Evaluation
```python
from src.metrics.spam_metrics import SpamMetrics
from src.metrics.conversation_metrics import ConversationMetrics

# Calculate metrics
spam_metrics = SpamMetrics()
conversation_metrics = ConversationMetrics()

# Evaluate results
spam_scores = spam_metrics.evaluate(predictions)
conversation_scores = conversation_metrics.evaluate(predictions)
```

## 📈 Performance Optimization

### 1. Caching
- Use appropriate cache directories
- Implement cache invalidation
- Monitor cache hit rates
- Optimize cache size

### 2. Batch Processing
- Process data in batches
- Implement parallel processing
- Monitor memory usage
- Optimize batch size

### 3. Model Optimization
- Tune model parameters
- Optimize prompts
- Monitor response times
- Implement retry mechanisms

## 🔍 Troubleshooting

### 1. Common Issues
- Cache directory permissions
- Model API rate limits
- Memory usage spikes
- Cross-platform compatibility

### 2. Debugging
- Check logs in `logs/`
- Use debug mode for detailed output
- Monitor performance metrics
- Validate data formats

## 📝 Best Practices

### 1. Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write clear docstrings
- Maintain consistent formatting

### 2. Testing
- Write unit tests for new features
- Include edge cases
- Test cross-platform compatibility
- Validate performance

### 3. Documentation
- Keep documentation up to date
- Include examples
- Document configuration options
- Maintain changelog

## 🎯 Future Development

### 1. Planned Features
- Advanced analytics
- Real-time processing
- Additional model providers
- Enhanced visualization

### 2. Performance Goals
- Reduce response times
- Optimize memory usage
- Improve accuracy
- Scale for large datasets

### 3. Documentation Goals
- API documentation
- User guides
- Performance guides
- Deployment guides

## 🤝 Contributing
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Run tests
5. Update documentation
6. Submit PR

## 📞 Support
- Check documentation
- Review logs
- Consult team members
- Report issues

Remember to keep this guide updated as the project evolves! 