"""
Tests for all model classes including base model, clustering model, and optimization model.
"""

import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.models.base_model import BaseModel
from src.models.clustering_model import ClusteringModel
from src.models.optimization_model import OptimizationModel

# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0]
    ])

@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return np.array([0, 0, 0, 1, 1, 2])

# Base Model Tests
def test_base_model_initialization():
    """Test base model initialization."""
    model = BaseModel()
    assert model.model is None
    assert model.is_fitted is False

def test_base_model_fit_predict(sample_data, sample_labels):
    """Test base model fit and predict methods."""
    model = BaseModel()
    model.fit(sample_data, sample_labels)
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data)
    assert model.is_fitted is True

def test_base_model_persistence(tmp_path, sample_data, sample_labels):
    """Test base model persistence."""
    model = BaseModel()
    model.fit(sample_data, sample_labels)
    
    # Save model
    save_path = tmp_path / "base_model.pkl"
    model.save(save_path)
    
    # Load model
    loaded_model = BaseModel.load(save_path)
    predictions1 = model.predict(sample_data)
    predictions2 = loaded_model.predict(sample_data)
    assert np.array_equal(predictions1, predictions2)

# Clustering Model Tests
def test_clustering_model_initialization():
    """Test clustering model initialization."""
    model = ClusteringModel()
    assert isinstance(model.model, DBSCAN)
    assert model.model.eps == 0.5
    assert model.model.min_samples == 5

def test_clustering_model_fit_predict(sample_data):
    """Test clustering model fit and predict methods."""
    model = ClusteringModel()
    model.fit(sample_data)
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data)
    assert model.is_fitted is True

def test_clustering_model_parameters():
    """Test clustering model parameter setting."""
    model = ClusteringModel(
        eps=0.3,
        min_samples=3,
        metric='euclidean'
    )
    assert model.model.eps == 0.3
    assert model.model.min_samples == 3
    assert model.model.metric == 'euclidean'

def test_clustering_model_noise_handling(sample_data):
    """Test clustering model noise handling."""
    model = ClusteringModel(eps=0.1)  # Small eps to create noise points
    model.fit(sample_data)
    predictions = model.predict(sample_data)
    assert -1 in predictions  # Check for noise points

# Optimization Model Tests
def test_optimization_model_initialization():
    """Test optimization model initialization."""
    model = OptimizationModel()
    assert isinstance(model.model, GridSearchCV)
    assert isinstance(model.model.estimator, RandomForestClassifier)

def test_optimization_model_parameter_grid():
    """Test optimization model parameter grid."""
    model = OptimizationModel()
    assert 'n_estimators' in model.model.param_grid
    assert 'max_depth' in model.model.param_grid
    assert 'min_samples_split' in model.model.param_grid

def test_optimization_model_fit_predict(sample_data, sample_labels):
    """Test optimization model fit and predict methods."""
    model = OptimizationModel()
    model.fit(sample_data, sample_labels)
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data)
    assert model.is_fitted is True

def test_optimization_model_best_params(sample_data, sample_labels):
    """Test optimization model best parameters."""
    model = OptimizationModel()
    model.fit(sample_data, sample_labels)
    assert hasattr(model.model, 'best_params_')
    assert isinstance(model.model.best_params_, dict)

def test_optimization_model_cross_validation(sample_data, sample_labels):
    """Test optimization model cross-validation."""
    model = OptimizationModel()
    model.fit(sample_data, sample_labels)
    assert hasattr(model.model, 'cv_results_')
    assert 'mean_test_score' in model.model.cv_results_

# Error Handling Tests
def test_models_empty_data():
    """Test handling of empty data."""
    models = [BaseModel(), ClusteringModel(), OptimizationModel()]
    empty_data = np.array([])
    
    for model in models:
        with pytest.raises(ValueError):
            model.fit(empty_data)

def test_models_invalid_data():
    """Test handling of invalid data."""
    models = [BaseModel(), ClusteringModel(), OptimizationModel()]
    invalid_data = "not an array"
    
    for model in models:
        with pytest.raises(ValueError):
            model.fit(invalid_data)

def test_models_predict_before_fit(sample_data):
    """Test prediction before fitting."""
    models = [BaseModel(), ClusteringModel(), OptimizationModel()]
    
    for model in models:
        with pytest.raises(ValueError):
            model.predict(sample_data) 