# Quick Start Guide 🚀

## Setting Up Poultry Weight Predictor

### 1. First Time Setup
```bash
# Clone repository
git clone https://github.com/bomino/PoultryPredict.git

# Navigate to project
cd PoultryPredict

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt

# Start application
streamlit run app/main.py
```

### 2. Prepare Your Data
Your CSV file should include:
```csv
Int Temp,Int Humidity,Air Temp,Wind Speed,Feed Intake,Weight
29.87,59,29.6,4.3,11.00,42.39
```

Required columns:
- `Int Temp`: Internal Temperature (°C)
- `Int Humidity`: Internal Humidity (%)
- `Air Temp`: Air Temperature (°C)
- `Wind Speed`: Wind Speed (m/s)
- `Feed Intake`: Feed Intake (g)
- `Weight`: Weight (g) - for training only

Data quality requirements:
- No missing values
- Numerical values only
- Values within reasonable ranges
- Consistent units
- Clean formatting

### 3. Using the Application

#### a. Data Upload & Analysis
1. Click "Browse files"
2. Select your CSV file
3. Review data preview and summary
4. Check data quality metrics
5. Analyze feature relationships
6. Review outlier detection results

#### b. Model Training
1. Select model type:
   - Polynomial Regression (for simple patterns)
   - Gradient Boosting (for complex patterns)
   - Support Vector Regression (for robust predictions)
2. Configure model parameters
3. Set test split size
4. Enable advanced options if needed:
   - Cross-validation
   - Early stopping (for Gradient Boosting)
5. Click "Train Model"
6. Review comprehensive results

#### c. Making Predictions
1. Choose input method:
   - Manual Input: Enter values directly
   - Batch Prediction: Upload CSV file
2. Enter/upload prediction data
3. Get predictions with confidence metrics
4. Validate results
5. Download predictions

#### d. Model Comparison
1. Train multiple models
2. Visit comparison page
3. Review detailed metrics
4. Compare feature importance
5. Export comprehensive report

### 4. Troubleshooting

Common issues and solutions:

1. **Data Upload Errors**
   - Check column names match exactly
   - Ensure all values are numeric
   - Remove any special characters
   - Verify data types
   - Check for hidden whitespace

2. **Training Errors**
   - Verify data quality
   - Check parameter ranges
   - Ensure sufficient data
   - Monitor for overfitting
   - Check system resources

3. **Prediction Errors**
   - Validate input ranges
   - Verify data format
   - Check model training status
   - Review feature consistency
   - Ensure proper scaling

4. **Model Performance Issues**
   - Start with simpler models
   - Tune hyperparameters
   - Use cross-validation
   - Enable early stopping
   - Check for data quality issues

### 5. Best Practices

1. **Data Preparation**
   - Clean your data thoroughly
   - Handle outliers appropriately
   - Use consistent units
   - Document preprocessing steps
   - Validate data ranges

2. **Model Selection**
   - Consider dataset size
   - Evaluate data complexity
   - Start with simpler models
   - Use model recommendations
   - Compare multiple approaches

3. **Model Training**
   - Start with default parameters
   - Use cross-validation
   - Monitor performance metrics
   - Document optimal settings
   - Save best models

4. **Making Predictions**
   - Stay within training ranges
   - Validate unusual results
   - Keep prediction logs
   - Monitor performance
   - Use confidence metrics

### 6. Getting Help

If you need assistance:
1. Check error messages
2. Review documentation
3. Check troubleshooting guide
4. Review best practices
5. Open an issue on GitHub

### 7. Quick Tips

1. **Data Quality**
   - Use the provided template
   - Check for outliers first
   - Review feature correlations
   - Validate ranges before training

2. **Model Selection**
   - Use Polynomial Regression for small datasets
   - Try Gradient Boosting for complex patterns
   - Use SVR when robustness is key
   - Compare multiple models

3. **Performance Optimization**
   - Enable cross-validation
   - Use early stopping
   - Monitor training metrics
   - Review feature importance
   - Document best configurations

---

For detailed information, see [Documentation.md](Documentation.md)