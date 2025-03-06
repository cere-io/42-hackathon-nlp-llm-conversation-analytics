# NLP/LLM Conversation Analytics - Interview Tasks

## Overview
This interview consists of two main components:
1. **AI/ML Tasks**: Focusing on Natural Language Processing, data cleaning, and vector similarity
2. **Software Engineering Tasks**: Focusing on code organization, containerization, and production-ready implementation

## 🤖 AI/ML Tasks
This section evaluates your understanding of NLP concepts and your ability to implement core ML functionality from scratch. You'll work with real conversation data to build a complete NLP pipeline.

<details>
<summary>1. Data Analysis and NLP - Parse and clean conversation data, implementing basic NLP operations from scratch</summary>

**Requirements:**
- Parse the dataset from: `data/groups/thisiscere/messages_thisiscere.csv`
- Implement data cleaning operations:
  - Handle missing values
  - Remove irrelevant columns
  - Format timestamps
  - Text cleaning (special characters, standardization)
  - Optional: Implement spam message detection and filtering
- Implement text tokenization from scratch:
  - Handle word boundaries
  - Process punctuation and special characters
  - Manage case sensitivity
  - Optional: stop words removal, contraction handling
- Document your NLP pipeline decisions
</details>

<details>
<summary>2. Text Vectorization - Transform text into numerical vectors for semantic analysis</summary>

**Requirements:**
- Implement text vectorization functionality
- Choose and implement an appropriate vectorization method
- Consider dimensionality of the output vectors

**Questions to Consider** ❓
- What vectorization approach would you choose for this use case?
- How would you handle the vocabulary and embedding dimensions?
- How would you evaluate the quality of your vector representations?
</details>

<details>
<summary>3. Vector Similarity Implementation - Design and implement vector similarity analysis for text comparison</summary>

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

**Questions to Consider** ❓
- Which similarity metric would be most appropriate for text vectors and why?
- How would you handle different vector dimensions?
- What are your options for vector storage and retrieval?
- How would you scale this for a large number of vectors?
</details>

## 💻 Software Engineering Tasks
This section evaluates your software engineering practices and ability to create production-ready code. Focus is on code quality, documentation, and deployment readiness.

<details>
<summary>1. Code Organization and Documentation - Create a well-structured, documented, and maintainable codebase</summary>

**Requirements:**
- Clear project structure
- Comprehensive README with:
  - Project overview
  - Setup instructions
  - Usage examples
- Well-documented code with:
  - Function documentation
  - Type hints
  - Inline comments for complex logic
- Dependencies management:
  - requirements.txt or environment.yml
  - Version specifications

**Questions to Consider** ❓
- How would you ensure your script runs consistently across different machines?
- What steps would you take to make the codebase maintainable?
- How would you handle version conflicts between dependencies?
</details>

<details>
<summary>2. Containerization and Deployment - Package the application for production deployment using Docker</summary>

**Requirements:**
- Docker implementation:
  - Efficient Dockerfile
  - Clear build instructions
  - Runtime configuration
- Environment handling:
  - Configuration management
  - Environment variables
  - Resource considerations
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
- Documentation completeness
- Docker implementation
- Production readiness
- Best practices adherence
</details>

## Submission Guidelines
Your final submission should demonstrate proficiency in both technical implementation and software engineering practices.

<details>
<summary>Submission Requirements - Complete list of required deliverables</summary>

1. **AI/ML Deliverables:**
   - Data processing pipeline
   - NLP implementation
   - Vectorization implementation
   - Vector similarity functions
   - Analysis documentation

2. **Software Engineering Deliverables:**
   - Complete codebase
   - Docker configuration
   - Documentation
   - Requirements specification 