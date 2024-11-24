# Poultry Weight Predictor Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Requirements](#data-requirements)
4. [Features and Functionality](#features-and-functionality)
5. [Technical Architecture](#technical-architecture)
6. [Machine Learning Models](#machine-learning-models)
7. [User Guide](#user-guide)
8. [Troubleshooting](#troubleshooting)
9. [Development Guide](#development-guide)
10. [API Reference](#api-reference)

## Overview

The Poultry Weight Predictor is a machine learning application built with Streamlit that helps poultry farmers and researchers predict poultry weight based on environmental and feeding data. The application provides comprehensive data analysis, model training, and prediction capabilities with support for multiple advanced machine learning models.

### Key Features
- Advanced data upload and validation
- Interactive data analysis and visualization
- Multiple machine learning model support
- Comprehensive outlier detection and analysis
- Model training and evaluation
- Real-time predictions
- Advanced model comparison and performance analysis
- Export capabilities for reports and predictions

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)
- Git (for version control)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/bomino/PoultryPredict.git
cd PoultryPredict
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Dependencies
```text
streamlit==1.32.0
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
plotly==5.18.0
pytest==8.0.0
python-dotenv==1.0.0
joblib==1.3.2
xlsxwriter==3.1.9
```

## Data Requirements

### Input Data Format
The application expects CSV files with the following columns:

| Column Name    | Description                | Unit  | Type    | Valid Range |
|---------------|---------------------------|-------|---------|-------------|
| Int Temp      | Internal Temperature      | °C    | float   | 15-40       |
| Int Humidity  | Internal Humidity         | %     | float   | 30-90       |
| Air Temp      | Air Temperature           | °C    | float   | 10-45       |
| Wind Speed    | Wind Speed               | m/s   | float   | 0-15        |
| Feed Intake   | Feed Intake              | g     | float   | > 0         |
| Weight        | Poultry Weight (target)  | g     | float   | > 0         |

### Data Quality Requirements
- No missing values
- Numerical values only
- Consistent units
- Values within reasonable ranges
- Minimum of 2 samples for training
- Clean data formatting

## Features and Functionality

### 1. Data Upload (Page 1)
- CSV file upload capability
- Automatic data validation
- Enhanced data preprocessing
- Data preview and summary statistics
- Comprehensive data quality checks
- Sample template download
- Data format verification

### 2. Data Analysis (Page 2)
- Time series analysis
- Feature relationship exploration
- Correlation analysis
- Multi-level outlier detection
- Comprehensive feature statistics
- Interactive visualizations
- Export capabilities for analyzed data

### 3. Model Training (Page 3)
- Multiple model types support:
  - Polynomial Regression
  - Gradient Boosting
  - Support Vector Regression
- Advanced hyperparameter tuning
- Cross-validation with configurable folds
- Model performance metrics
- Feature importance analysis
- Early stopping capabilities
- Model saving and loading
- Training metadata tracking

### 4. Predictions (Page 4)
- Single prediction through manual input
- Batch predictions via CSV upload
- Prediction history tracking
- Confidence intervals
- Export capabilities
- Input validation
- Prediction accuracy monitoring

### 5. Model Comparison (Page 5)
- Side-by-side model comparison
- Comprehensive metrics comparison
- Feature importance comparison
- Export comparison reports
- Visual performance analysis
- Best model recommendations

## Machine Learning Models

### 1. Polynomial Regression
- **Description**: Non-linear regression using polynomial features
- **Parameters**:
  - degree: Polynomial degree (default: 2)
  - include_bias: Include bias term (default: True)
  - fit_intercept: Fit intercept (default: True)
- **Strengths**:
  - Good for capturing non-linear patterns
  - Simple and interpretable
  - Works well with small datasets
  - Fast training and prediction
- **Limitations**:
  - May overfit with high polynomial degrees
  - Sensitive to outliers
  - Limited complexity handling

### 2. Gradient Boosting
- **Description**: Ensemble learning using gradient boosting
- **Parameters**:
  - n_estimators: Number of boosting stages
  - learning_rate: Learning rate
  - max_depth: Maximum tree depth
  - min_samples_split: Minimum samples for split
  - min_samples_leaf: Minimum samples in leaf
  - subsample: Subsample ratio
  - early_stopping_rounds: Early stopping iterations
  - validation_fraction: Validation set size
- **Strengths**:
  - Handles non-linear relationships well
  - Robust to outliers
  - High accuracy
  - Built-in feature importance
- **Limitations**:
  - More computationally intensive
  - Requires more tuning
  - Can overfit if not properly configured

### 3. Support Vector Regression
- **Description**: Kernel-based regression for robust predictions
- **Parameters**:
  - kernel: Kernel type ('rbf', 'linear', 'poly')
  - C: Regularization parameter
  - epsilon: Epsilon in epsilon-SVR model
  - gamma: Kernel coefficient
  - cache_size: Kernel cache size
- **Strengths**:
  - Excellent generalization
  - Robust to outliers
  - Handles non-linear relationships
  - Works well with medium-sized datasets
- **Limitations**:
  - Slower training on large datasets
  - Memory intensive
  - Requires careful kernel selection
  - Less intuitive feature importance

## User Guide

### Getting Started
1. **Data Preparation**
   - Prepare CSV file with required columns
   - Ensure data quality requirements are met
   - Use provided template if needed
   - Validate data ranges and formats

2. **Data Upload**
   - Navigate to "Data Upload" page
   - Upload prepared CSV file
   - Review data summary and quality checks
   - Address any validation errors
   - Explore data preview

3. **Data Analysis**
   - Examine data distributions
   - Analyze feature relationships
   - Review correlation analysis
   - Perform outlier detection
   - Export analyzed data if needed

4. **Model Training**
   - Select appropriate model type
   - Configure model parameters
   - Set training/test split ratio
   - Enable/configure cross-validation
   - Train model and review metrics
   - Save trained model if desired

5. **Making Predictions**
   - Choose prediction method
   - Input or upload prediction data
   - Get and validate predictions
   - Export results
   - Track prediction history

6. **Model Comparison**
   - Train multiple models
   - Compare performance metrics
   - Analyze feature importance differences
   - Select best performing model
   - Export comparison report

### Best Practices
1. **Data Quality**
   - Clean data thoroughly before upload
   - Handle outliers appropriately
   - Ensure consistent units
   - Validate data ranges
   - Document any data preprocessing

2. **Model Selection**
   - Consider dataset size
   - Evaluate data complexity
   - Start with simpler models
   - Use model recommendations
   - Compare multiple approaches

3. **Parameter Tuning**
   - Start with default parameters
   - Use cross-validation
   - Monitor for overfitting
   - Enable early stopping when appropriate
   - Document optimal configurations

## API Reference

### DataProcessor Class
```python
class DataProcessor:
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame
    def prepare_features(self, df: pd.DataFrame, test_size: float) -> tuple
    def validate_columns(self, df: pd.DataFrame) -> tuple[bool, list]
    def scale_features(self, X: pd.DataFrame) -> np.ndarray
    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.Series
```

### BaseModel Class Methods
```python
def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None)
def predict(self, X: np.ndarray) -> np.ndarray
def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]
def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]
def save(self, filepath: str)
@classmethod
def load(cls, filepath: str)
```

### ModelFactory Class
```python
class ModelFactory:
    def get_model(self, model_type: str, params: dict = None) -> BaseModel
    def get_available_models(self) -> dict
    def get_model_params(self, model_type: str) -> dict
    def get_param_descriptions(self, model_type: str) -> dict
```

### Visualizer Class
```python
class Visualizer:
    def plot_correlation_matrix(self, df: pd.DataFrame)
    def plot_feature_importance(self, feature_names: list, importance_values: list)
    def plot_actual_vs_predicted(self, y_true: list, y_pred: list)
    def plot_residuals(self, y_true: list, y_pred: list)
    def plot_feature_distribution(self, df: pd.DataFrame, column: str)
```

For detailed API documentation, refer to the docstrings in each module.