from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

class PoultryGBRegressor:
    """
    Gradient Boosting Regressor for poultry weight prediction with enhanced functionality.
    Includes early stopping, feature importance analysis, and model persistence.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the Gradient Boosting model.
        
        Args:
            params (Optional[Dict]): Model parameters. If None, uses default parameters.
        """
        # Default model parameters
        self.default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'random_state': 42
        }
        
        # Initialize parameters
        if params is not None:
            # Extract early stopping parameters if present
            self.early_stopping_rounds = params.pop('early_stopping_rounds', None)
            self.validation_fraction = params.pop('validation_fraction', 0.1)
            # Merge remaining parameters with defaults
            self.params = {**self.default_params, **params}
        else:
            self.early_stopping_rounds = None
            self.validation_fraction = 0.1
            self.params = self.default_params.copy()
        
        # Initialize model and state
        self.model = GradientBoostingRegressor(**self.params)
        self._is_trained = False
        self.feature_names_ = None
        self.feature_importances_ = None
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
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None) -> 'PoultryGBRegressor':
        """
        Train the model with optional early stopping.
        
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
            
            print(f"Training Gradient Boosting model with {len(X_train)} samples...")
            
            # Implementation of early stopping if enabled
            if self.early_stopping_rounds is not None:
                self._train_with_early_stopping(X_train, y_train)
            else:
                self.model.fit(X_train, y_train)
            
            # Store feature importances and mark as trained
            self.feature_importances_ = self.model.feature_importances_
            self._is_trained = True
            
            # Store training metadata
            self.training_metadata = {
                'n_samples': len(X_train),
                'n_features': X_train.shape[1],
                'training_date': datetime.now().isoformat(),
                'parameters': self.params,
                'early_stopping_used': self.early_stopping_rounds is not None
            }
            
            print("Model training completed successfully")
            return self
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def _train_with_early_stopping(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Implement early stopping training procedure.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        # Split data for validation
        n_samples = len(X_train)
        n_val = int(n_samples * self.validation_fraction)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train_sub = X_train[train_indices]
        y_train_sub = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        best_val_score = float('-inf')
        best_model = None
        patience_counter = 0
        
        for n_est in range(1, self.params['n_estimators'] + 1):
            # Train model with current number of estimators
            current_params = {**self.params, 'n_estimators': n_est}
            current_model = GradientBoostingRegressor(**current_params)
            current_model.fit(X_train_sub, y_train_sub)
            
            # Evaluate on validation set
            val_score = current_model.score(X_val, y_val)
            
            if val_score > best_val_score:
                best_val_score = val_score
                best_model = current_model
                patience_counter = 0
                print(f"New best validation score: {best_val_score:.4f} at iteration {n_est}")
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_rounds:
                print(f"Early stopping triggered at iteration {n_est}")
                break
        
        self.model = best_model
    
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
        Get feature importance scores.
        
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
            # Use provided feature names, stored names, or generate default names
            if feature_names is None:
                feature_names = self.feature_names_
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]
            
            # Create and sort importance dictionary
            importance_dict = dict(zip(feature_names, self.feature_importances_))
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
                'feature_importances': self.feature_importances_,
                'is_trained': self._is_trained,
                'early_stopping_rounds': self.early_stopping_rounds,
                'validation_fraction': self.validation_fraction,
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
    def load(cls, filepath: str) -> 'PoultryGBRegressor':
        """
        Load a saved model from a file.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            PoultryGBRegressor: Loaded model instance
            
        Raises:
            FileNotFoundError: If model file is not found
            Exception: If loading fails
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            save_dict = joblib.load(filepath)
            
            # Create new instance with saved parameters
            params = save_dict['params'].copy()
            params['early_stopping_rounds'] = save_dict['early_stopping_rounds']
            params['validation_fraction'] = save_dict['validation_fraction']
            instance = cls(params=params)
            
            # Restore saved state
            instance.model = save_dict['model']
            instance.feature_names_ = save_dict['feature_names']
            instance.feature_importances_ = save_dict['feature_importances']
            instance._is_trained = save_dict['is_trained']
            instance.training_metadata = save_dict['training_metadata']
            
            return instance
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def get_model_params(self) -> Dict:
        """Get current model parameters."""
        return self.params