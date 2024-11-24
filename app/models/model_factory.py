from typing import Dict, Optional, Any
import os
import sys

# Add the parent directory to sys.path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import models
from models.polynomial_regression import PoultryWeightPredictor
from models.gradient_boosting import PoultryGBRegressor
from models.svr_model import PoultrySVR

class ModelFactory:
    """Enhanced Model Factory with SVR support."""
    
    @staticmethod
    def get_model(model_type: str, params: Optional[Dict] = None) -> Any:
        """
        Create and return a model instance.
        
        Args:
            model_type: Type of model to create
            params: Optional model parameters
        """
        models = {
            'polynomial': PoultryWeightPredictor,
            'gradient_boosting': PoultryGBRegressor,
            'svr': PoultrySVR
        }
        
        if model_type.lower() not in models:
            raise ValueError(f"Unknown model type: {model_type}. Available models: {list(models.keys())}")
            
        return models[model_type.lower()](params)
    
    @staticmethod
    def get_available_models() -> Dict:
        """Get list of available models with descriptions."""
        return {
            'polynomial': {
                'name': 'Polynomial Regression',
                'description': 'Traditional polynomial regression for baseline predictions.',
                'strengths': [
                    'Simple and interpretable',
                    'Fast training and prediction',
                    'Good for basic non-linear relationships',
                    'Low computational requirements'
                ],
                'limitations': [
                    'May overfit with high polynomial degrees',
                    'Sensitive to outliers',
                    'Limited complexity handling',
                    'Requires careful feature scaling'
                ],
                'use_cases': [
                    'Initial baseline modeling',
                    'Simple trend analysis',
                    'When interpretability is crucial'
                ]
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'description': 'Advanced ensemble learning model for complex patterns.',
                'strengths': [
                    'Handles non-linear relationships well',
                    'Robust to outliers',
                    'Provides feature importance',
                    'High prediction accuracy'
                ],
                'limitations': [
                    'More computationally intensive',
                    'Requires more hyperparameter tuning',
                    'Can overfit if not properly configured',
                    'Less interpretable than simpler models'
                ],
                'use_cases': [
                    'Complex pattern recognition',
                    'When high accuracy is needed',
                    'Large dataset handling'
                ]
            },
            'svr': {
                'name': 'Support Vector Regression',
                'description': 'Advanced kernel-based regression for robust predictions.',
                'strengths': [
                    'Excellent generalization',
                    'Robust to outliers',
                    'Handles non-linear relationships well',
                    'Works well with medium-sized datasets'
                ],
                'limitations': [
                    'Slower training on large datasets',
                    'Requires careful kernel selection',
                    'Memory intensive for large datasets',
                    'Less intuitive feature importance'
                ],
                'use_cases': [
                    'Robust regression needs',
                    'When polynomial regression overfits',
                    'Medium-sized datasets'
                ]
            }
        }
    
    @staticmethod
    def get_model_params(model_type: str) -> Dict:
        """Get default parameters for each model type."""
        params = {
            'polynomial': {
                'degree': {
                    'default': 2,
                    'range': (1, 5),
                    'type': 'int',
                    'description': 'Polynomial degree for feature transformation'
                }
            },
            'gradient_boosting': {
                'n_estimators': {
                    'default': 100,
                    'range': (50, 500),
                    'type': 'int',
                    'description': 'Number of boosting stages'
                },
                'learning_rate': {
                    'default': 0.1,
                    'range': (0.01, 0.3),
                    'type': 'float',
                    'description': 'Learning rate shrinks the contribution of each tree'
                },
                'max_depth': {
                    'default': 3,
                    'range': (2, 10),
                    'type': 'int',
                    'description': 'Maximum depth of the individual trees'
                },

                'validation_fraction': {
                    'default': 0.1,
                    'range': (0.1, 0.3),
                    'type': 'float',
                    'description': 'Fraction of training data to use for early stopping'
                },
                'early_stopping_rounds': {
                    'default': 10,
                    'range': (5, 50),
                    'type': 'int',
                    'description': 'Number of rounds with no improvement before early stopping'
                }
            },
            'svr': {
                'kernel': {
                    'default': 'rbf',
                    'options': ['rbf', 'linear', 'poly'],
                    'type': 'select',
                    'description': 'Kernel type for non-linear relationships'
                },
                'C': {
                    'default': 1.0,
                    'range': (0.1, 10.0),
                    'type': 'float',
                    'description': 'Regularization parameter'
                },
                'epsilon': {
                    'default': 0.1,
                    'range': (0.01, 1.0),
                    'type': 'float',
                    'description': 'Epsilon in the epsilon-SVR model'
                },
                'gamma': {
                    'default': 'scale',
                    'options': ['scale', 'auto'],
                    'type': 'select',
                    'description': 'Kernel coefficient'
                }
            }
        }
        
        if model_type.lower() not in params:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return params[model_type.lower()]
    
    @staticmethod
    def get_param_descriptions(model_type: str) -> Dict[str, str]:
        """Get parameter descriptions for a specific model type."""
        params = ModelFactory.get_model_params(model_type)
        return {param: info['description'] for param, info in params.items()}


    @staticmethod
    def get_model_comparison_metrics() -> Dict:
        """Define metrics for model comparison."""
        return {
            'mse': {
                'name': 'Mean Squared Error',
                'lower_is_better': True,
                'format': '.4f'
            },
            'rmse': {
                'name': 'Root Mean Squared Error',
                'lower_is_better': True,
                'format': '.4f'
            },
            'r2': {
                'name': 'R² Score',
                'lower_is_better': False,
                'format': '.4f'
            },
            'mae': {
                'name': 'Mean Absolute Error',
                'lower_is_better': True,
                'format': '.4f'
            },
            'mape': {
                'name': 'Mean Absolute Percentage Error',
                'lower_is_better': True,
                'format': '.2f'
            }
        }

    @staticmethod
    def suggest_model(data_characteristics: Dict) -> str:
        """Suggest best model based on data characteristics."""
        n_samples = data_characteristics.get('n_samples', 0)
        n_features = data_characteristics.get('n_features', 0)
        has_outliers = data_characteristics.get('has_outliers', False)
        complexity = data_characteristics.get('complexity', 'medium')
        
        if n_samples < 100:
            return 'polynomial'  # Simple model for small datasets
        elif has_outliers and n_samples < 1000:
            return 'svr'  # Robust to outliers
        else:
            return 'gradient_boosting'  # Complex patterns, large datasets