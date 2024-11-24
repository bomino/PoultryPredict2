# Poultry Weight Predictor Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Requirements](#data-requirements)
4. [Features and Functionality](#features-and-functionality)
5. [Machine Learning Models](#machine-learning-models)
6. [Model Recommendation System](#model-recommendation-system)
7. [User Guide](#user-guide)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

## Overview

The Poultry Weight Predictor is a machine learning application built with Streamlit that helps poultry farmers and researchers predict poultry weight based on environmental and feeding data. The application provides comprehensive data analysis, model training, and prediction capabilities.

### Key Features
- Data upload and validation
- Interactive data analysis and visualization
- Multiple machine learning model support
- Intelligent model recommendation
- Model training and evaluation
- Real-time predictions
- Model comparison and performance analysis
- Export capabilities for reports and predictions

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

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

## Data Requirements

### Input Data Format
The application expects CSV files with the following columns:

| Column Name    | Description                | Unit  | Type    | Valid Range |
|---------------|---------------------------|-------|---------|-------------|
| Int Temp      | Internal Temperature      | °C    | float   | 15-40      |
| Int Humidity  | Internal Humidity         | %     | float   | 30-90      |
| Air Temp      | Air Temperature           | °C    | float   | 10-45      |
| Wind Speed    | Wind Speed               | m/s   | float   | 0-15       |
| Feed Intake   | Feed Intake              | g     | float   | > 0        |
| Weight        | Poultry Weight (target)  | g     | float   | > 0        |

### Data Quality Requirements
- No missing values
- Numerical values only
- Values within specified ranges
- Consistent units
- Minimum of 2 samples for training

## Features and Functionality

### 1. Data Upload and Validation
- CSV file upload support
- Automatic data validation
- Data preview and summary statistics
- Data quality assessment
- Sample template download

### 2. Data Analysis
- Time series analysis of weight progression
- Feature relationship exploration
- Correlation analysis
- Advanced outlier detection
- Interactive visualizations
- Statistical summaries
- Data distribution analysis

### 3. Model Training
- Multiple model types supported:
  - Polynomial Regression
  - Gradient Boosting
  - Support Vector Regression
- Interactive parameter tuning
- Cross-validation support
- Feature importance analysis
- Model performance metrics
- Model saving capabilities

### 4. Predictions
- Single prediction through manual input
- Batch predictions via CSV upload
- Prediction history tracking
- Confidence metrics
- Result export capabilities

### 5. Model Comparison
- Side-by-side model comparison
- Performance metrics comparison
- Feature importance comparison
- Prediction accuracy analysis
- Export comparison reports

## Machine Learning Models

### 1. Polynomial Regression
- **Description**: Non-linear regression using polynomial features
- **Parameters**:
  - degree: Polynomial degree (default: 2)
  - include_bias: Include bias term
  - fit_intercept: Fit intercept
- **Best for**:
  - Small datasets
  - Simple patterns
  - Baseline modeling
  - High interpretability needs

### 2. Gradient Boosting
- **Description**: Ensemble learning using gradient boosting
- **Parameters**:
  - n_estimators: Number of boosting stages
  - learning_rate: Learning rate
  - max_depth: Maximum tree depth
  - early_stopping_rounds: Early stopping iterations
- **Best for**:
  - Large datasets
  - Complex patterns
  - High accuracy needs
  - Feature importance analysis

### 3. Support Vector Regression
- **Description**: Kernel-based regression for robust predictions
- **Parameters**:
  - kernel: Kernel type ('rbf', 'linear', 'poly')
  - C: Regularization parameter
  - epsilon: Epsilon in epsilon-SVR model
  - gamma: Kernel coefficient
- **Best for**:
  - Medium-sized datasets
  - Outlier presence
  - Non-linear relationships
  - Robust predictions

## Model Recommendation System

### Analysis Process
The system analyzes dataset characteristics to recommend the most appropriate model:
- Sample size
- Presence of outliers
- Data complexity
- Feature relationships
- Training requirements

### Recommendation Logic
1. **Small Datasets** (< 100 samples)
   - Recommends: Polynomial Regression
   - Reasoning: Better generalization with limited data

2. **Medium Datasets with Outliers**
   - Recommends: Support Vector Regression
   - Reasoning: Robust to outliers, good generalization

3. **Large Datasets** (> 1000 samples)
   - Recommends: Gradient Boosting
   - Reasoning: Handles complex patterns, high accuracy

### Usage Guidelines
1. Review data analysis results
2. Consider recommendation reasoning
3. Evaluate alternative models if needed
4. Monitor model performance

## User Guide

### Getting Started
1. **Data Preparation**
   - Prepare CSV file with required columns
   - Ensure data quality requirements are met
   - Use template if needed

2. **Data Upload**
   - Navigate to "Data Upload" page
   - Upload CSV file
   - Review data summary
   - Check validation results

3. **Data Analysis**
   - Explore data distributions
   - Check feature relationships
   - Review outlier analysis
   - Analyze correlations

4. **Model Training**
   - Review model recommendation
   - Select model type
   - Configure parameters
   - Train and evaluate
   - Save model if desired

5. **Making Predictions**
   - Choose prediction method
   - Input or upload data
   - Get predictions
   - Export results

### Best Practices
1. **Data Quality**
   - Clean data thoroughly
   - Handle outliers appropriately
   - Use consistent units
   - Validate ranges

2. **Model Selection**
   - Consider dataset characteristics
   - Follow recommendations
   - Compare multiple models
   - Document settings

3. **Model Training**
   - Use cross-validation
   - Monitor metrics
   - Avoid overfitting
   - Save best models

## API Reference

### Core Classes
```python
class DataProcessor:
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame
    def prepare_features(self, df: pd.DataFrame, test_size: float) -> tuple
    def validate_columns(self, df: pd.DataFrame) -> tuple[bool, list]
    def scale_features(self, X: pd.DataFrame) -> np.ndarray

class ModelFactory:
    def get_model(self, model_type: str, params: dict = None)
    def get_available_models(self) -> dict
    def get_model_params(self, model_type: str) -> dict
    def get_param_descriptions(self, model_type: str) -> dict

class BaseModel:
    def train(self, X_train: np.ndarray, y_train: np.ndarray)
    def predict(self, X: np.ndarray) -> np.ndarray
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict
    def get_feature_importance(self) -> dict
```

## Troubleshooting

### Common Issues and Solutions

1. **Data Upload Errors**
- **Problem**: Invalid format/columns
  - **Solution**: Check CSV format, column names
- **Problem**: Data validation failures
  - **Solution**: Review data quality requirements

2. **Model Training Issues**
- **Problem**: Poor performance
  - **Solution**: Try recommended model, adjust parameters
- **Problem**: Training errors
  - **Solution**: Check data quality, reduce complexity

3. **Prediction Issues**
- **Problem**: Unreasonable predictions
  - **Solution**: Validate input ranges
- **Problem**: Low confidence
  - **Solution**: Retrain model, check data quality
