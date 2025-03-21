# NLP/LLM Conversation Analytics - Interview Tasks

## Overview
This interview consists of three main components:
1. **Data Science Tasks**: Focusing on data exploration, cleaning, and preprocessing
2. **AI/ML Tasks**: Focusing on text vectorization and similarity analysis
3. **Software Engineering Tasks**: Focusing on code organization, production readiness, and deployment

While completing these tasks successfully demonstrates excellent technical capability, we're equally interested in understanding your problem-solving approach, thinking process, and working habits throughout the interview. 😊

## 📊 Data Science Tasks
This section evaluates your ability to work with real data, handle data cleaning, and prepare text for analysis.

<details>
<summary>1. Data Analysis and Preprocessing - Explore and clean conversation data</summary>

**Requirements:**
- Parse the dataset from: [`data/groups/thisiscere/messages_thisiscere.csv`](data/groups/thisiscere/messages_thisiscere.csv)
- Implement data cleaning operations:
  - Handle missing values
  - Remove irrelevant columns
  - Format timestamps
  - Text cleaning (special characters, standardization)
  - Optional: Implement spam message detection and filtering
- Implement text tokenization:
  - Handle word boundaries
  - Process punctuation and special characters
  - Manage case sensitivity
  - Optional: stop words removal, contraction handling
- Document your implementation decisions

---
> **💭 Questions to Consider**
> - How would you approach exploring this dataset?
> - What data quality issues would you look for?
> - How would you handle edge cases in the text data?
> - What limitations do you see in this dataset and how could it be improved?
</details>

## 🤖 AI/ML Tasks
This section evaluates your understanding of NLP concepts and ability to implement ML functionality.

<details>
<summary>1. Text Vectorization - Transform text into numerical vectors for semantic analysis</summary>

**Requirements:**
- Implement text vectorization functionality
- Choose and implement an appropriate vectorization method
- Consider dimensionality of the output vectors

---
> **💭 Questions to Consider**
> - What vectorization approach would you choose for this use case?
> - How would you handle the vocabulary and embedding dimensions?
> - How would you evaluate the quality of your vector representations?
</details>

<details>
<summary>2. Vector Similarity Implementation - Design and implement vector similarity analysis for text comparison</summary>

**Requirements:**
- Implement vector similarity analysis
- Function specifications:
  - Vector input handling
  - Edge case management
  - Similarity score calculation
- Optional enhancements:
  - Explore different similarity metrics
  - Comparative analysis of approaches
  - Performance optimization

---
> **💭 Questions to Consider**
> - Which similarity metric would be most appropriate for text vectors and why?
> - How would you handle different vector dimensions?
> - What are your options for vector storage and retrieval?
> - How would you scale this for a large number of vectors?
</details>

## 💻 Software Engineering Tasks
This section evaluates your software engineering practices and ability to create production-ready code. Focus is on code quality, documentation, and deployment readiness.

<details>
<summary>1. Code Organization and Documentation - Create a well-structured and readable codebase</summary>

**Requirements:**
- Clear code organization
- Basic README with essential setup and usage info
- Clean code practices:
  - Clear function names
  - Basic documentation
  - Helpful comments for complex logic
- Basic dependency handling
- Comprehensive test suite organized by functionality:
  - Models: Base, clustering, and optimization model tests
  - Detectors: Spam and conversation detection tests
  - Processors: Text vectorization and parallel processing tests
  - Metrics: Clustering and conversation evaluation tests
  - CLI: Command-line interface tests

---
> **💭 Questions to Consider**
> - How would you ensure your script runs consistently across different machines?
> - What steps would you take to make the codebase maintainable?
> - How would you handle version conflicts between dependencies?
> - How would you organize tests to maximize coverage and maintainability?
> - What testing strategies would you employ for different components?
</details>

<details>
<summary>2. Production Deployment - Package the application for production use</summary>

**Requirements:**
- Production-ready configuration
- Environment handling:
  - Configuration management
  - Environment variables
  - Resource considerations

---
> **💭 Questions to Consider**
> - How would you handle sensitive configuration data in a production environment?
> - What monitoring and logging strategies would you implement?
</details>

## Evaluation Criteria
Your submission will be evaluated based on both technical implementation and software engineering best practices.

<details>
<summary>🤖 AI/ML Evaluation - Assessment of NLP implementation and machine learning concepts</summary>

- Understanding of NLP concepts
- Data cleaning methodology
- Vectorization approach
- Vector similarity implementation
- Algorithm efficiency
- Edge case handling in ML context
</details>

<details>
<summary>💻 Software Engineering Evaluation - Assessment of code quality and production readiness</summary>

- Code quality and organization
- Documentation clarity
- Deployment strategy
- Production readiness
- Best practices adherence
</details> 