from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Tuple, List, Optional
from datetime import datetime

class PoultrySVR:
    """Support Vector Regression model for poultry weight prediction."""
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the SVR model with optimized defaults for poultry weight prediction.
        
        Args:
            params: Optional dictionary of model parameters
        """
        self.default_params = {
            'kernel': 'rbf',        # Radial basis function kernel
            'C': 1.0,              # Regularization parameter
            'epsilon': 0.1,        # Epsilon in the epsilon-SVR model
            'gamma': 'scale',      # Kernel coefficient
            'tol': 1e-3,          # Tolerance for stopping criterion
            'cache_size': 500,     # Kernel cache size (MB)
            'max_iter': -1,        # No limit on iterations
            'verbose': False,      # Output debug information
            'random_state': 42     # For reproducibility
        }
        
        self.params = {**self.default_params, **(params or {})}
        self.model = SVR(**self.params)
        self.scaler = StandardScaler()
        self._is_trained = False
        self.feature_names_ = None
        
    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Train the SVR model with scaled features.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Optional list of feature names
            
        Returns:
            Self for method chaining
        """
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        
        try:
            # Store feature names if provided
            if feature_names is not None:
                self.feature_names_ = feature_names

            # Scale features
            print("Scaling features...")
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            print(f"Training SVR with {X_train.shape[0]} samples...")
            self.model.fit(X_scaled, y_train)
            self._is_trained = True
            
            # Store training metadata
            self.training_metadata = {
                'n_samples': X_train.shape[0],
                'n_features': X_train.shape[1],
                'n_support_vectors': len(self.model.support_vectors_),
                'support_vector_ratio': len(self.model.support_vectors_) / X_train.shape[0]
            }
            
            print(f"Training completed. Support vectors: {self.training_metadata['n_support_vectors']} "
                  f"({self.training_metadata['support_vector_ratio']:.2%} of training data)")
            
            return self
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with scaled features.
        """
        if not self._is_trained:
            raise ValueError("Model needs to be trained before prediction")
            
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            X_test: Test features
            y_test: True test values
            
        Returns:
            Tuple of (metrics_dict, predictions)
        """
        if not self._is_trained:
            raise ValueError("Model needs to be trained before evaluation")
            
        try:
            # Get predictions
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': mean_squared_error(y_test, y_pred, squared=False),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            # Add detailed error analysis
            error = np.abs(y_test - y_pred)
            metrics.update({
                'max_error': np.max(error),
                'min_error': np.min(error),
                'std_error': np.std(error),
                'q90_error': np.percentile(error, 90)
            })
            
            return metrics, y_pred
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Estimate feature importance using model coefficients or weights.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of feature importances
        """
        if not self._is_trained:
            raise ValueError("Model needs to be trained before calculating feature importance")
            
        try:
            # Use stored feature names if none provided
            if feature_names is None:
                feature_names = self.feature_names_ or [f'feature_{i}' for i in range(self.training_metadata['n_features'])]
                
            # Calculate importance based on kernel weights
            if self.params['kernel'] == 'linear':
                # For linear kernel, use coefficients directly
                importance_scores = np.abs(self.model.coef_[0])
            else:
                # For non-linear kernels, use support vector weights
                importance_scores = np.sum(np.abs(self.model.dual_coef_[0]), axis=0)
            
            # Normalize scores
            importance_scores = importance_scores / np.sum(importance_scores)
            
            # Create importance dictionary
            importance_dict = dict(zip(feature_names, importance_scores))
            
            # Sort by importance
            return dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            raise
    
    def save(self, filepath: str):
        """Save the model with all components."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before saving")
            
        try:
            save_dict = {
                'model': self.model,
                'scaler': self.scaler,
                'params': self.params,
                'feature_names': self.feature_names_,
                'is_trained': self._is_trained,
                'training_metadata': getattr(self, 'training_metadata', None),
                'save_timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(save_dict, filepath)
            print(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'PoultrySVR':
        """Load a saved model with validation."""
        try:
            save_dict = joblib.load(filepath)
            
            instance = cls(params=save_dict['params'])
            instance.model = save_dict['model']
            instance.scaler = save_dict['scaler']
            instance.feature_names_ = save_dict['feature_names']
            instance._is_trained = save_dict['is_trained']
            instance.training_metadata = save_dict.get('training_metadata', {})
            
            return instance
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
    def get_model_params(self):
        """Get the current model parameters."""
        return self.params