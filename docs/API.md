# API Documentation

## Overview
This document provides detailed information about the Conversation Analytics System API.

## Core Components

### Models

#### BaseModel
```python
class BaseModel:
    """
    Base class for all models in the system.
    
    Methods:
        fit(data: np.ndarray) -> None
        predict(data: np.ndarray) -> np.ndarray
        save(path: str) -> None
        load(path: str) -> None
    """
```

#### ClusteringModel
```python
class ClusteringModel(BaseModel):
    """
    Model for clustering text vectors.
    
    Parameters:
        n_clusters (int): Number of clusters
        algorithm (str): Clustering algorithm to use
        random_state (int): Random seed
        
    Methods:
        fit(data: np.ndarray) -> None
        predict(data: np.ndarray) -> np.ndarray
        get_cluster_centers() -> np.ndarray
    """
```

#### OptimizationModel
```python
class OptimizationModel(BaseModel):
    """
    Model for optimizing clustering parameters.
    
    Parameters:
        param_grid (dict): Parameter grid for optimization
        cv (int): Number of cross-validation folds
        
    Methods:
        fit(data: np.ndarray) -> None
        predict(data: np.ndarray) -> np.ndarray
        get_best_params() -> dict
        get_best_score() -> float
    """
```

### Detectors

#### ConversationDetector
```python
class ConversationDetector:
    """
    Detects conversations in message sequences.
    
    Parameters:
        time_threshold (float): Maximum time gap between messages
        similarity_threshold (float): Minimum similarity for grouping
        max_conversation_size (int): Maximum messages per conversation
        
    Methods:
        detect(messages: List[dict], vectors: np.ndarray) -> List[dict]
        get_conversation_stats() -> dict
    """
```

#### SpamDetector
```python
class SpamDetector:
    """
    Detects spam messages in text.
    
    Parameters:
        threshold (float): Classification threshold
        model_type (str): Type of model to use
        
    Methods:
        fit(texts: List[str], labels: List[int]) -> None
        predict(texts: List[str]) -> np.ndarray
        get_feature_importance() -> dict
    """
```

### Processors

#### TextVectorizer
```python
class TextVectorizer:
    """
    Vectorizes text for analysis.
    
    Parameters:
        max_features (int): Maximum number of features
        cache_dir (str): Directory for caching results
        
    Methods:
        fit(texts: List[str]) -> None
        transform(texts: List[str]) -> np.ndarray
        get_vocabulary() -> dict
    """
```

#### ParallelProcessor
```python
class ParallelProcessor:
    """
    Processes data in parallel.
    
    Parameters:
        n_jobs (int): Number of parallel jobs
        batch_size (int): Size of processing batches
        
    Methods:
        process(data: List, func: callable) -> List
        process_batch(data: List, func: callable) -> List
    """
```

### Metrics

#### ConversationMetrics
```python
class ConversationMetrics:
    """
    Calculates conversation-related metrics.
    
    Methods:
        evaluate(conversations: List[dict]) -> dict
        calculate_duration_stats(conversations: List[dict]) -> dict
        calculate_participant_stats(conversations: List[dict]) -> dict
    """
```

#### ClusteringMetrics
```python
class ClusteringMetrics:
    """
    Calculates clustering-related metrics.
    
    Methods:
        evaluate(data: np.ndarray, labels: np.ndarray) -> dict
        calculate_silhouette(data: np.ndarray, labels: np.ndarray) -> float
        calculate_calinski_harabasz(data: np.ndarray, labels: np.ndarray) -> float
    """
```

## CLI Interface

### Commands

#### Pre-group Messages
```bash
python -m conversation_analytics pre-group --input <input_file> --output <output_dir>
```

#### Evaluate Results
```bash
python -m conversation_analytics evaluate --directory <results_dir>
```

#### Optimize Models
```bash
python -m conversation_analytics optimize --input <input_dir> --output <output_dir>
```

## Configuration

### Environment Variables
- `CACHE_DIR`: Directory for caching results
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_WORKERS`: Maximum number of parallel workers
- `MODEL_CACHE_DIR`: Directory for model caching

### Configuration Files
- `config.yaml`: Main configuration file
- `model_config.yaml`: Model-specific configuration
- `optimization_config.yaml`: Optimization parameters

## Error Handling

### Common Exceptions
- `ValueError`: Invalid input data or parameters
- `FileNotFoundError`: Missing required files
- `RuntimeError`: Processing errors
- `MemoryError`: Insufficient memory

### Error Codes
- 1: Invalid arguments
- 2: File not found
- 3: Processing error
- 4: Memory error
- 5: Configuration error 