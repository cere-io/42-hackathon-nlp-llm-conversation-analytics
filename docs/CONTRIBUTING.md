# Contributing Guidelines

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- A GitHub account

### Setting Up Development Environment
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/conversation-analytics.git
   cd conversation-analytics
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Process

### 1. Create a New Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-fix-name
```

### 2. Make Changes
- Follow the coding style guidelines
- Write clear commit messages
- Add tests for new features
- Update documentation as needed

### 3. Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/models/
python -m pytest tests/detectors/
python -m pytest tests/processors/
python -m pytest tests/metrics/
python -m pytest tests/cli/

# Run with coverage
python -m pytest --cov=src tests/
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add new feature"
# or
git commit -m "fix: resolve issue"
```

### 5. Push Changes
```bash
git push origin feature/your-feature-name
```

## Code Style

### Python Style Guide
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Documentation Style
- Keep documentation up to date
- Use clear and concise language
- Include examples where appropriate
- Follow the established format

## Pull Request Process

### 1. Update Documentation
- Update README.md if needed
- Update API documentation
- Add comments for complex code

### 2. Create Pull Request
- Use the PR template
- Provide a clear description
- Link related issues
- Include test results

### 3. Code Review
- Address review comments
- Make requested changes
- Keep commits clean and focused

## Testing Guidelines

### Unit Tests
- Test each component independently
- Cover edge cases
- Mock external dependencies
- Use appropriate fixtures

### Integration Tests
- Test component interactions
- Verify data flow
- Check error handling
- Validate performance

## Documentation Guidelines

### Code Documentation
- Use clear and concise docstrings
- Include parameter descriptions
- Document return values
- Note any exceptions

### API Documentation
- Document all public interfaces
- Provide usage examples
- Include parameter types
- Document error cases

## Release Process

### Versioning
- Follow semantic versioning
- Update CHANGELOG.md
- Tag releases in Git
- Update version in setup.py

### Deployment
- Test in staging environment
- Verify all tests pass
- Check documentation
- Create release notes

## Getting Help

### Resources
- Check existing documentation
- Review closed issues
- Ask in discussions
- Contact maintainers

### Reporting Issues
- Use issue templates
- Provide reproduction steps
- Include error messages
- Add system information

## License
By contributing, you agree that your contributions will be licensed under the MIT License. 