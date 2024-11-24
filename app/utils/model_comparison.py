import pandas as pd
import numpy as np
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px

class ModelComparison:
    def __init__(self):
        self.results = {}
        self.predictions = {}
        
    def add_model_results(self, model_name: str, metrics: Dict[str, float], 
                         predictions: np.ndarray, actual: np.ndarray,
                         feature_importance: Dict[str, float] = None):
        """Add results for a model to the comparison."""
        self.results[model_name] = {
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        self.predictions[model_name] = {
            'predicted': np.array(predictions),  # Ensure numpy array
            'actual': np.array(actual)          # Ensure numpy array
        }
    
    def get_metrics_comparison(self) -> pd.DataFrame:
        """Get a DataFrame comparing metrics across models."""
        metrics_data = {}
        for model_name, result in self.results.items():
            metrics_data[model_name] = result['metrics']
        return pd.DataFrame(metrics_data).round(4)
    
    def get_prediction_comparison(self) -> pd.DataFrame:
        """Get a DataFrame comparing predictions across models."""
        # First, create a base DataFrame with the actual values
        first_model = list(self.predictions.keys())[0]
        base_df = pd.DataFrame({
            'Sample': range(1, len(self.predictions[first_model]['actual']) + 1),
            'Actual': self.predictions[first_model]['actual']
        })
        
        # Add predictions and errors for each model
        for model_name, pred_data in self.predictions.items():
            predicted = pred_data['predicted']
            actual = pred_data['actual']
            
            # Calculate errors
            abs_error = np.abs(actual - predicted)
            rel_error = np.abs((actual - predicted) / actual) * 100
            
            # Add to DataFrame
            base_df[f'Predicted_{model_name}'] = predicted
            base_df[f'Absolute_Error_{model_name}'] = abs_error
            base_df[f'Relative_Error_%_{model_name}'] = rel_error
        
        return base_df
    
    def plot_metrics_comparison(self, metric: str = 'r2'):
        """Create a bar plot comparing a specific metric across models."""
        metrics_df = self.get_metrics_comparison()
        
        if metric not in metrics_df.index:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {metrics_df.index.tolist()}")
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_df.columns,
                y=metrics_df.loc[metric],
                text=metrics_df.loc[metric].round(4),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f'{metric.upper()} Score Comparison',
            xaxis_title='Models',
            yaxis_title=f'{metric.upper()} Score',
            template='plotly_white'
        )
        
        return fig
    
    def plot_prediction_comparison(self):
        """Create a scatter plot comparing predictions across models."""
        fig = go.Figure()
        
        # Find global min and max for the perfect prediction line
        all_actuals = np.concatenate([pred_data['actual'] for pred_data in self.predictions.values()])
        all_predictions = np.concatenate([pred_data['predicted'] for pred_data in self.predictions.values()])
        min_val = min(np.min(all_actuals), np.min(all_predictions))
        max_val = max(np.max(all_actuals), np.max(all_predictions))
        
        # Add perfect prediction line
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='black', dash='dash')
        ))
        
        # Add model predictions
        for model_name, pred_data in self.predictions.items():
            fig.add_trace(go.Scatter(
                x=pred_data['actual'],
                y=pred_data['predicted'],
                mode='markers',
                name=model_name,
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Actual vs Predicted Values Comparison',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def plot_feature_importance_comparison(self):
        """Create a heatmap comparing feature importance across models."""
        # Collect feature importance data
        importance_data = {}
        for model_name, result in self.results.items():
            if result['feature_importance'] is not None:
                importance_data[model_name] = result['feature_importance']
        
        if not importance_data:
            return None
            
        # Create DataFrame
        importance_df = pd.DataFrame(importance_data)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=importance_df.values,
            x=importance_df.columns,
            y=importance_df.index,
            colorscale='Viridis',
            text=np.round(importance_df.values, 4),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Importance Comparison',
            xaxis_title='Models',
            yaxis_title='Features',
            template='plotly_white'
        )
        
        return fig
    
    def get_best_model(self, metric: str = 'r2') -> str:
        """Get the name of the best performing model based on a specific metric."""
        metrics_df = self.get_metrics_comparison()
        if metric not in metrics_df.index:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {metrics_df.index.tolist()}")
            
        # Handle metrics where lower is better
        lower_is_better = metric.lower() in ['mse', 'rmse', 'mae']
        if lower_is_better:
            return metrics_df.loc[metric].idxmin()
        return metrics_df.loc[metric].idxmax()
    
    def get_model_rankings(self, metric: str = 'r2') -> pd.Series:
        """Get models ranked by a specific metric."""
        metrics_df = self.get_metrics_comparison()
        if metric not in metrics_df.index:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {metrics_df.index.tolist()}")
            
        # Handle metrics where lower is better
        lower_is_better = metric.lower() in ['mse', 'rmse', 'mae']
        return metrics_df.loc[metric].sort_values(ascending=lower_is_better)
    
    def export_comparison_report(self) -> Dict[str, Any]:
        """Export a comprehensive comparison report."""
        metrics_df = self.get_metrics_comparison()
        
        return {
            'metrics_comparison': metrics_df,
            'prediction_comparison': self.get_prediction_comparison(),
            'model_rankings': {
                metric: self.get_model_rankings(metric).to_dict()
                for metric in metrics_df.index
            },
            'best_models': {
                metric: self.get_best_model(metric)
                for metric in metrics_df.index
            }
        }