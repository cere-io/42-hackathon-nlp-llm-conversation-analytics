"""
Clustering model for conversation detection.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class ClusteringModel(BaseModel):
    """
    Clustering model for detecting conversations using DBSCAN.
    
    This model uses DBSCAN clustering to group messages into conversations
    based on their semantic similarity and temporal proximity.
    """
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 2,
        metric: str = 'cosine',
        params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize the clustering model.
        
        Args:
            eps: Maximum distance between samples for DBSCAN
            min_samples: Minimum number of samples in a neighborhood
            metric: Distance metric for DBSCAN
            params: Optional dictionary of additional parameters
            random_state: Optional random seed
        """
        super().__init__(params, random_state)
        
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric
        )
        
        logger.info(
            f"Initialized ClusteringModel with eps={eps}, "
            f"min_samples={min_samples}, metric={metric}"
        )
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ClusteringModel':
        """
        Fit the clustering model.
        
        Args:
            X: Training data
            y: Not used, kept for compatibility
            
        Returns:
            The fitted model instance
        """
        self.validate_data(X)
        
        logger.info("Fitting clustering model")
        self.model.fit(X)
        self.is_fitted = True
        
        # Calculate silhouette score if possible
        if len(np.unique(self.model.labels_)) > 1:
            score = silhouette_score(X, self.model.labels_)
            logger.info(f"Silhouette score: {score:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data to predict labels for
            
        Returns:
            Array of cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_data(X)
        
        # Use fit_predict to handle new data
        labels = self.model.fit_predict(X)
        return labels
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get the size of each cluster.
        
        Returns:
            Dictionary mapping cluster labels to sizes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster sizes")
        
        unique_labels, counts = np.unique(self.model.labels_, return_counts=True)
        return dict(zip(unique_labels, counts))
    
    def get_noise_points(self) -> np.ndarray:
        """
        Get indices of noise points (points labeled as -1).
        
        Returns:
            Array of noise point indices
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting noise points")
        
        return np.where(self.model.labels_ == -1)[0]
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get the center of each cluster.
        
        Returns:
            Array of cluster centers
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster centers")
        
        # Calculate mean of points in each cluster
        centers = []
        for label in np.unique(self.model.labels_):
            if label != -1:  # Skip noise points
                cluster_points = X[self.model.labels_ == label]
                centers.append(np.mean(cluster_points, axis=0))
        
        return np.array(centers) 