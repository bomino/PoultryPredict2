# Poultry Weight Predictor 🐔

A sophisticated machine learning application built with Streamlit for predicting poultry weight based on environmental and feeding data. This tool helps poultry farmers and researchers make data-driven decisions using advanced machine learning models and comprehensive analysis tools.

## ✨ Features

### 📊 Data Management
- **Smart Upload System**
  - CSV file upload with validation
  - Automatic data type detection
  - Quality assessment
  - Error detection and reporting
  - Template generation

- **Advanced Analysis**
  - Comprehensive outlier detection
  - Feature correlation analysis
  - Time series visualization
  - Statistical summaries
  - Data quality metrics

### 🤖 Machine Learning
- **Multiple Models Support**
  - Polynomial Regression (for simple patterns)
  - Gradient Boosting (for complex patterns)
  - Support Vector Regression (for robust predictions)

- **Intelligent Model Selection**
  - Automated model recommendations
  - Data-driven suggestions
  - Performance-based guidance
  - Optimization tips

- **Training Features**
  - Cross-validation
  - Feature importance analysis
  - Performance monitoring
  - Early stopping
  - Model persistence

### 📈 Predictions & Analysis
- **Flexible Prediction Options**
  - Single-value predictions
  - Batch processing
  - Real-time updates
  - Confidence metrics

- **Interactive Visualizations**
  - Performance comparisons
  - Feature importance charts
  - Error analysis
  - Time series plots

### 🔄 Model Comparison
- **Comprehensive Metrics**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²) Score
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)

- **Visual Comparisons**
  - Side-by-side analysis
  - Performance charts
  - Feature importance comparison
  - Error distribution

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager
- Virtual environment (recommended)

### Installation
```bash
# Clone repository
git clone https://github.com/bomino/PoultryPredict2.git
cd PoultryPredict

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt

# Launch application
streamlit run app/main.py
```

## 📋 Data Requirements

Your CSV file needs these columns:
```csv
Int Temp,Int Humidity,Air Temp,Wind Speed,Feed Intake,Weight
29.5,65,28.0,3.2,120,1500
```

### Required Columns
| Column       | Description       | Unit | Range    |
|-------------|------------------|------|----------|
| Int Temp    | House Temperature | °C   | 15-40    |
| Int Humidity| House Humidity    | %    | 30-90    |
| Air Temp    | Outside Temp      | °C   | 10-45    |
| Wind Speed  | Wind Speed       | m/s  | 0-15     |
| Feed Intake | Feed Consumed    | g    | > 0      |
| Weight      | Poultry Weight   | g    | > 0      |

## 💡 Usage Guide

### 1. Data Upload
- Upload your CSV file
- Review validation results
- Check data quality metrics
- Address any issues flagged

### 2. Data Analysis
- Explore data distributions
- Check feature relationships
- Identify outliers
- Review correlations
- Analyze time series patterns

### 3. Model Training
- Review model recommendations
- Select appropriate model
- Configure parameters
- Train and validate
- Compare performance

### 4. Making Predictions
- Choose prediction method
- Input or upload data
- Get predictions with confidence
- Export results

## 🛠️ Advanced Features

### Model Selection System
The application analyzes your data and recommends the best model based on:
- Dataset size
- Presence of outliers
- Data complexity
- Feature relationships
- Performance requirements

### Cross-Validation
- Configurable validation splits
- Performance metrics across folds
- Stability assessment
- Overfitting detection

### Feature Importance
- Feature ranking
- Impact analysis
- Visual representation
- Comparative analysis

## 📚 Documentation

For detailed information, see:
- [Documentation.md](Documentation.md) - Comprehensive guide
- [QUICK_START.md](QUICK_START.md) - Getting started guide

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## 💪 Best Practices

### Data Preparation
- Clean your data thoroughly
- Remove obvious outliers
- Ensure consistent units
- Validate value ranges

### Model Selection
- Start with recommended model
- Compare multiple approaches
- Monitor performance
- Document settings

### Making Predictions
- Validate input data
- Check confidence metrics
- Keep prediction logs
- Monitor accuracy

## 🆘 Support

Need help? Check:
- Documentation
- Issue tracker
- Feature requests
- Email: Bomino@mlawali.com

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with Streamlit
- Uses scikit-learn
- Plotly for visualizations
- Pandas for data handling

---

Made with ❤️ for poultry farmers and researchers

![GitHub stars](https://img.shields.io/github/stars/bomino/PoultryPredict2?style=social)
![GitHub forks](https://img.shields.io/github/forks/bomino/PoultryPredict2?style=social)
