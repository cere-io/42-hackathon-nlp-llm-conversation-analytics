# English and Project Alignment Rule

## Language Guidelines
- All code, comments, and documentation MUST be written in English 
- Variable names, function names, and class names should use clear English terms
- Log messages and user-facing strings should be in English
- Commit messages should be in English

## Project Alignment
All code changes must align with:
###powershell terminal guiding steps###
### @README.md Requirements
- Data processing and evaluation capabilities
- Model integration (Mistral, GPT-4, etc.)
- Metrics calculation and reporting
- File structure and naming conventions

### @interview_tasks.md Criteria
1. Data Science Tasks:
   - Data exploration and cleaning
   - Text preprocessing
   - Spam detection
   - Tokenization

2. AI/ML Tasks:
   - Text vectorization
   - Vector similarity analysis
   - Performance optimization
   - Edge case handling

3. Software Engineering Tasks:
   - Code organization
   - Documentation
   - Production readiness
   - Environment handling

## Code Style
- Use descriptive English names that reflect purpose
- Follow Python naming conventions:
  - Classes: PascalCase (e.g., `BatchProcessor`, `ScalableProcessor`)
  - Functions/Methods: snake_case (e.g., `process_messages`, `vectorize_texts`)
  - Variables: snake_case (e.g., `batch_size`, `max_memory_mb`)
  - Constants: UPPER_SNAKE_CASE (e.g., `MAX_BATCH_SIZE`, `DEFAULT_CACHE_DIR`)

## Documentation
- All docstrings must be in English
- Follow this format for function/method documentation:
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function purpose.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description of when this exception is raised
    """
```

## Implementation Checklist
- [ ] Code uses English throughout
- [ ] Documentation is in English
- [ ] Aligns with README.md requirements:
  - [ ] Data processing functionality
  - [ ] Model integration
  - [ ] Metrics reporting
  - [ ] Environment configuration
- [ ] Meets interview_tasks.md criteria:
  - [ ] Data Science tasks completed
  - [ ] AI/ML tasks implemented
  - [ ] Software Engineering best practices followed
- [ ] Follows Python naming conventions
- [ ] Includes proper English docstrings
- [ ] Error handling and logging in English
- [ ] Configuration and environment variables documented 