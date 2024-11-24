# Poultry Weight Predictor 🐔

A sophisticated machine learning application built with Streamlit for predicting poultry weight based on environmental and feeding data. This tool helps poultry farmers and researchers make data-driven decisions using multiple advanced machine learning models and comprehensive analysis tools.

## Features

### 1. Data Management
- **Data Upload**:
  - CSV file upload support
  - Automated data validation and preprocessing
  - Dynamic data type handling
  - Missing value detection
  - Column validation
  - Range checking

- **Data Preprocessing**:
  - Automatic data cleaning
  - Feature scaling
  - Multi-level outlier detection
  - Comprehensive data validation
  - Feature correlation analysis
  - Statistical analysis

### 2. Model Training
- **Multiple Models Support**:
  - Polynomial Regression (for baseline and simple patterns)
  - Gradient Boosting Regressor (for complex patterns)
  - Support Vector Regression (for robust predictions)
  - Extensible architecture for future models

- **Model Configuration**:
  - Dynamic parameter settings
  - Model-specific parameter validation
  - Interactive parameter tuning
  - Cross-validation support
  - Early stopping capabilities
  - Performance metrics tracking
  - Training metadata capture

- **Training Features**:
  - Configurable train/test split
  - Advanced cross-validation
  - Feature importance analysis
  - Performance visualization
  - Model persistence
  - Training history

### 3. Model Comparison
- **Comparison Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²) Score
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)
  - Feature importance comparison
  - Cross-validation scores

- **Visualization Tools**:
  - Interactive performance plots
  - Feature importance charts
  - Prediction comparison graphs
  - Error analysis visualizations
  - Residual plots
  - Correlation heatmaps

### 4. Prediction Capabilities
- **Flexible Input Methods**:
  - Single prediction through UI
  - Batch predictions via CSV
  - Real-time prediction updates
  - Input validation
  - Range checking

- **Output Features**:
  - Detailed prediction analysis
  - Confidence metrics
  - Error estimates
  - Exportable results
  - Prediction history tracking
  - Performance monitoring

## Installation

### Prerequisites
- Python 3.9+
- pip package manager
- Virtual environment (recommended)
- Git (for version control)

### Setup

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

4. Run the application:
```bash
streamlit run app/main.py
```

## Project Structure
```
poultry_weight_predictor/
│
├── app/
│   ├── main.py                  # Application entry point
│   │
│   ├── pages/
│   │   ├── 1_Data_Upload.py       # Data upload and validation
│   │   ├── 2_Data_Analysis.py     # Data analysis and visualization
│   │   ├── 3_Model_Training.py    # Model training and evaluation
│   │   ├── 4_Predictions.py       # Prediction interface
│   │   ├── 5_Model_Comparison.py  # Model comparison tools
│   │   └── 6_About.py            # About and documentation
│   │
│   ├── models/
│   │   ├── model_factory.py       # Model creation and management
│   │   ├── polynomial_regression.py# Polynomial regression model
│   │   ├── gradient_boosting.py   # Gradient boosting model
│   │   └── svr_model.py          # Support Vector Regression model
│   │
│   ├── utils/
│   │   ├── data_processor.py      # Data processing utilities
│   │   ├── visualizations.py      # Visualization functions
│   │   └── model_comparison.py    # Model comparison utilities
│   │
│   └── config/
│       └── settings.py            # Application settings
│
├── models/                      # Saved models directory
├── tests/                      # Unit tests
├── Documentation.md            # Detailed documentation
├── QUICK_START.md             # Quick start guide
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Usage Guide

### 1. Data Preparation
Required CSV format:
```csv
Int Temp,Int Humidity,Air Temp,Wind Speed,Feed Intake,Weight
29.87,59,29.6,4.3,11.00,42.39
32.50,47,30.5,4.3,12.47,45.67
```

### 2. Model Training
1. Upload data in the Data Upload page
2. Analyze data quality and features
3. Navigate to Model Training
4. Select appropriate model type
5. Configure parameters
6. Train model
7. Review results and save model

### 3. Making Predictions
- **Single Prediction**:
  1. Enter values manually
  2. Validate inputs
  3. Get instant prediction
  4. Review confidence metrics

- **Batch Prediction**:
  1. Upload CSV file
  2. Validate data
  3. Get predictions for all rows
  4. Download results

### 4. Model Comparison
1. Train multiple models
2. Visit Model Comparison page
3. Compare performance metrics
4. Analyze feature importance
5. Export comparison report

## Configuration

### Model Parameters

1. Polynomial Regression:
   - Degree
   - Include bias
   - Fit intercept

2. Gradient Boosting:
   - Number of estimators
   - Learning rate
   - Maximum depth
   - Minimum samples split
   - Early stopping rounds
   - Validation fraction

3. Support Vector Regression:
   - Kernel type
   - Regularization parameter (C)
   - Epsilon
   - Gamma
   - Cache size

## Best Practices

1. **Data Quality**:
   - Clean data thoroughly
   - Handle outliers appropriately
   - Maintain consistent units
   - Validate ranges

2. **Model Selection**:
   - Start with simpler models
   - Use recommendations
   - Compare multiple approaches
   - Consider dataset size

3. **Training Process**:
   - Use cross-validation
   - Monitor metrics
   - Enable early stopping
   - Document settings

4. **Prediction Workflow**:
   - Validate inputs
   - Monitor accuracy
   - Keep prediction logs
   - Review confidence metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Streamlit
- Uses scikit-learn for machine learning
- Plotly for visualizations
- Pandas for data handling
- NumPy for numerical operations
- Joblib for model persistence

## Support

For support:
- Check the documentation
- Review troubleshooting guide
- Open an issue
- Contact: Bomino@mlawali.com

---

Made with ❤️ for poultry farmers and researchers