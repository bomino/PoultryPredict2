from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from config.settings import POLYNOMIAL_DEGREE

class PoultryWeightPredictor:
    """
    Polynomial Regression model for poultry weight prediction.
    Includes feature engineering, model evaluation, and persistence functionality.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the Polynomial Regression model.
        
        Args:
            params (Optional[Dict]): Model parameters. If None, uses default parameters.
        """
        # Default model parameters
        self.default_params = {
            'degree': POLYNOMIAL_DEGREE,
            'include_bias': True,
            'fit_intercept': True
        }
        
        # Initialize parameters
        self.params = {**self.default_params, **(params or {})}
        
        # Create the pipeline with polynomial features
        self.model = Pipeline([
            ('poly', PolynomialFeatures(
                degree=self.params['degree'],
                include_bias=self.params['include_bias']
            )),
            ('regressor', LinearRegression(
                fit_intercept=self.params['fit_intercept']
            ))
        ])
        
        # Initialize state variables
        self._is_trained = False
        self.feature_names_ = None
        self.poly_feature_names_ = None
        self.coefficients_ = None
        self.training_metadata = {}
    
    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained
    
    def _validate_input_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, is_training: bool = True):
        """
        Validate input data for training or prediction.
        
        Args:
            X: Input features
            y: Target values (optional)
            is_training: Whether this is for training data
            
        Raises:
            ValueError: If data validation fails
        """
        if X is None:
            raise ValueError("Input features cannot be None")
        if len(X) == 0:
            raise ValueError("Input features cannot be empty")
        if is_training:
            if y is None:
                raise ValueError("Target values cannot be None for training")
            if len(X) != len(y):
                raise ValueError("Number of samples in features and target must match")
            if len(X) < 2:
                raise ValueError("Need at least 2 samples for training")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> 'PoultryWeightPredictor':
        """
        Train the polynomial regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Optional list of feature names
            
        Returns:
            self: The trained model instance
            
        Raises:
            ValueError: If input validation fails
            Exception: If training fails
        """
        try:
            # Validate input data
            self._validate_input_data(X_train, y_train, is_training=True)
            
            # Store feature names if provided
            if feature_names is not None:
                self.feature_names_ = feature_names
            
            print(f"Training Polynomial Regression model with {len(X_train)} samples...")
            
            # Fit the model
            self.model.fit(X_train, y_train)
            
            # Get polynomial feature names
            if self.feature_names_ is not None:
                self.poly_feature_names_ = self.model.named_steps['poly'].get_feature_names_out(self.feature_names_)
            else:
                self.poly_feature_names_ = self.model.named_steps['poly'].get_feature_names_out()
                
            # Store coefficients
            self.coefficients_ = self.model.named_steps['regressor'].coef_
            
            # Mark as trained
            self._is_trained = True
            
            # Store training metadata
            self.training_metadata = {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'n_poly_features': len(self.poly_feature_names_),
                'training_date': datetime.now().isoformat(),
                'parameters': self.params,
                'r2_train': self.model.score(X_train, y_train)
            }
            
            print("Model training completed successfully")
            return self
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted values
            
        Raises:
            ValueError: If model is not trained or input validation fails
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self._validate_input_data(X, is_training=False)
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Evaluate model performance with multiple metrics.
        
        Args:
            X_test: Test features
            y_test: True test values
            
        Returns:
            Tuple containing:
                - Dictionary of evaluation metrics
                - Array of predicted values
                
        Raises:
            ValueError: If model is not trained or input validation fails
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            self._validate_input_data(X_test, y_test, is_training=False)
            y_pred = self.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': mean_squared_error(y_test, y_pred, squared=False),
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            return metrics, y_pred
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get feature importance based on coefficient magnitudes.
        
        Args:
            feature_names: Optional list of feature names to use
            
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
            
        Raises:
            ValueError: If model is not trained
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        try:
            # Use stored polynomial feature names or generate them
            if feature_names is not None and len(feature_names) == len(self.coefficients_):
                poly_features = feature_names
            else:
                poly_features = self.poly_feature_names_
            
            if poly_features is None:
                poly_features = [f'feature_{i}' for i in range(len(self.coefficients_))]
            
            # Calculate absolute importance
            importance = np.abs(self.coefficients_)
            
            # Normalize importance scores
            importance = importance / np.sum(importance)
            
            # Create and sort importance dictionary
            importance_dict = dict(zip(poly_features, importance))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            raise
    
    def save(self, filepath: str):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path where to save the model
            
        Raises:
            ValueError: If model is not trained
            Exception: If saving fails
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            save_dict = {
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names_,
                'poly_feature_names': self.poly_feature_names_,
                'coefficients': self.coefficients_,
                'is_trained': self._is_trained,
                'training_metadata': self.training_metadata,
                'save_timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(save_dict, filepath)
            print(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'PoultryWeightPredictor':
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            PoultryWeightPredictor: Loaded model instance
            
        Raises:
            FileNotFoundError: If model file is not found
            Exception: If loading fails
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            save_dict = joblib.load(filepath)
            
            # Create new instance with saved parameters
            instance = cls(params=save_dict['params'])
            
            # Restore saved state
            instance.model = save_dict['model']
            instance.feature_names_ = save_dict['feature_names']
            instance.poly_feature_names_ = save_dict['poly_feature_names']
            instance.coefficients_ = save_dict['coefficients']
            instance._is_trained = save_dict['is_trained']
            instance.training_metadata = save_dict['training_metadata']
            
            return instance
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def get_model_params(self) -> Dict:
        """Get current model parameters."""
        return self.params