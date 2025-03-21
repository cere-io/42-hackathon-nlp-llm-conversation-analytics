# Conversation Detection Optimization Tools

This directory contains a set of tools for optimizing conversation detection models through prompt engineering, pre-grouping techniques, and systematic experimentation.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Available Tools](#available-tools)
- [Complete Workflow](#complete-workflow)
- [Advanced Usage](#advanced-usage)

## Overview

These tools are designed to enhance the performance of conversation detection models by:

1. **Pre-grouping messages** based on temporal patterns, user interactions, and semantic similarity
2. **Optimizing prompts** based on model performance feedback
3. **Systematically experimenting** with different model parameters and configurations
4. **Orchestrating the complete workflow** to find optimal configurations

## Getting Started

### Prerequisites

Ensure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

Additional requirements for the optimization tools:

```bash
pip install scikit-learn
```

### Basic Usage

The easiest way to get started is to use the orchestration script, which will run the entire optimization workflow:

```bash
python optimize_conversation_detection.py data/groups/thisiscere/messages_thisiscere.csv
```

This will:
1. Apply pre-grouping techniques to the messages
2. Optimize prompts based on existing metrics
3. Run experiments with different configurations
4. Find the best configuration and run a final model

## Available Tools

### Pre-grouping Techniques

The `pre_grouping_techniques.py` script applies various grouping techniques to prepare messages for model processing:

```bash
python pre_grouping_techniques.py data/groups/thisiscere/messages_thisiscere.csv
```

Options:
- `--output`: Specify output file path (default: pre_grouped_messages_thisiscere.csv)
- `--time-window`: Time window in minutes for time-based grouping (default: 30)

### Prompt Optimization

The `prompt_optimization.py` script enhances prompts based on model performance:

```bash
python prompt_optimization.py open_source_examples/prompts/conversation_detection_prompt.txt data/groups/thisiscere/metrics_conversations_thisiscere.csv
```

Options:
- `--output`: Path to write the enhanced prompt
- `--results-dir`: Directory containing the result files (default: data/groups/thisiscere)

### Model Experimentation

The `experiment_models.py` script systematically tests different model configurations:

```bash
python experiment_models.py data/groups/thisiscere/messages_thisiscere.csv
```

Options:
- `--config`: Base configuration file (default: open_source_examples/model_config.yaml)
- `--output`: Directory to store experiment results (default: experiments)

## Complete Workflow

The `optimize_conversation_detection.py` script orchestrates the complete workflow:

```bash
python optimize_conversation_detection.py data/groups/thisiscere/messages_thisiscere.csv
```

Options:
- `--config`: Base configuration file (default: open_source_examples/model_config.yaml)
- `--template`: Base prompt template (default: open_source_examples/prompts/conversation_detection_prompt.txt)
- `--output-dir`: Directory to store optimization results (default: results)
- `--skip-pre-grouping`: Skip pre-grouping step
- `--skip-prompt-optimization`: Skip prompt optimization step
- `--skip-experimentation`: Skip model experimentation step
- `--time-window`: Time window in minutes for pre-grouping (default: 30)

## Advanced Usage

### Custom Prompt Templates

You can create your own prompt templates by modifying existing ones. The key requirements for a valid prompt template are:

1. Include the `[MESSAGES]` placeholder where the input messages should be inserted
2. Specify the expected output format (CSV with message_id, conversation_id, topic, timestamp, confidence)
3. Provide clear instructions for grouping messages into conversations

Example enhanced prompt template:

```
open_source_examples/prompts/CD_prompt_mejorado.txt
```

### Configuration Options

The model configuration file (`model_config.yaml`) supports the following sections:

1. **model**: Model selection and parameters
   - `names`: List of models to use (e.g., "deepseek:latest")
   - `temperature`: Controls randomness (lower = more deterministic)
   - `max_tokens`: Maximum response length
   - `top_p`: Sampling parameter (lower = more focused)

2. **prompt**: Prompt configuration
   - `path`: Path to the prompt template file

3. **processing**: Message processing configuration
   - `batch_size`: Number of messages to process at once
   - `max_context_messages`: Maximum number of context messages
   - `min_confidence_threshold`: Minimum confidence score to keep

4. **gpu**: GPU configuration
   - `enabled`: Whether to use GPU
   - `auto_select`: Auto-select GPU
   - `fallback_to_cpu`: Fallback to CPU if GPU unavailable

### Experiment Parameter Space

You can modify the parameter space explored in experiments by editing the `parameter_space` dictionary in `experiment_models.py`:

```python
parameter_space = {
    "model": {
        "temperature": [0.1, 0.2, 0.3],
        "top_p": [0.85, 0.9, 0.95]
    },
    "processing": {
        "batch_size": [4, 6, 8],
        "max_context_messages": [8, 10, 12]
    },
    "prompt": {
        "path": [
            "open_source_examples/prompts/conversation_detection_prompt.txt",
            "open_source_examples/prompts/CD_prompt_mejorado.txt",
            "open_source_examples/prompts/CD_prompt_alejandro_1.txt",
            "open_source_examples/prompts/CD_prompt_alejandro_2.txt"
        ]
    }
}
```

## Best Practices

1. **Start with the orchestration script** to get a baseline of performance
2. **Review the experiment results** to understand which parameters have the most impact
3. **Focus on prompt optimization** when model performance is inconsistent
4. **Use pre-grouping techniques** when dealing with large message volumes
5. **Experiment with different models** to find the best one for your specific data

## Troubleshooting

If you encounter issues:

1. **Check dependencies**: Ensure all required packages are installed
2. **Verify file paths**: Make sure all file paths are correct
3. **Check model availability**: Ensure the specified models are available in Ollama
4. **GPU memory**: If using GPU, check available memory with `nvidia-smi`
5. **Reduce batch size**: If experiencing memory issues, reduce the batch size 