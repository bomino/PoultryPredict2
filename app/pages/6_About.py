import streamlit as st

def app():
    st.title("ℹ️ About")
    
    st.markdown("""
    # Poultry Weight Predictor

    This application helps poultry farmers and researchers predict poultry weight based on environmental 
    and feeding data using advanced machine learning techniques and comprehensive model comparison capabilities.

    ## Features

    1. **Data Upload and Analysis**
       - Upload CSV files with poultry data
       - Automatic data validation and preprocessing
       - Interactive data visualization
       - Comprehensive statistical analysis
       - Multi-level outlier detection and analysis
       - Advanced data quality assessment

    2. **Advanced Analytics**
       - Time series analysis of weight progression
       - Feature relationship exploration
       - Multi-dimensional correlation analysis
       - Pattern detection and visualization
       - Comprehensive outlier analysis across features
       - Interactive data exploration tools

    3. **Machine Learning Models**
       - Multiple model support:
         * Polynomial Regression (for baseline linear and non-linear patterns)
         * Gradient Boosting (for complex pattern recognition)
         * Support Vector Regression (for robust predictions)
       - Automated feature importance analysis
       - Model persistence and versioning
       - Cross-validation capabilities
       - Early stopping for appropriate models

    4. **Model Training and Evaluation**
       - Interactive parameter tuning
       - Real-time performance metrics
       - Feature importance visualization
       - Advanced error analysis
       - Model saving and loading functionality
       - Comprehensive training metadata tracking

    5. **Predictions**
       - Single prediction through manual input
       - Batch predictions through CSV upload
       - Prediction history tracking
       - Confidence intervals
       - Downloadable prediction results
       - Performance monitoring
       - Prediction validation

    6. **Model Comparison**
       - Side-by-side model performance comparison
       - Comparative metrics visualization
       - Feature importance comparison across models
       - Prediction accuracy analysis
       - Detailed performance metrics:
         * Mean Squared Error (MSE)
         * Root Mean Squared Error (RMSE)
         * R² Score
         * Mean Absolute Error (MAE)
         * Mean Absolute Percentage Error (MAPE)
       - Exportable comparison reports
       - Visual performance charts
       - Best model recommendation based on data characteristics

    ## How to Use

    1. **Data Upload**: Start by uploading your CSV file in the Data Upload page
    2. **Data Analysis**: Explore and analyze your data with interactive visualizations
    3. **Model Training**: Train different models with optimized parameters
    4. **Model Comparison**: Compare models to select the best performer
    5. **Predictions**: Make predictions using your chosen model
    6. **Export Results**: Download predictions and comparison reports

    ## Data Requirements

    Your input data should contain the following features:
    - Internal Temperature (°C)
    - Internal Humidity (%)
    - Air Temperature (°C)
    - Wind Speed (m/s)
    - Feed Intake (g)
    - Weight (g) - required for training data only

    ## Model Details

    ### Polynomial Regression
    - Captures non-linear relationships
    - Good for baseline predictions
    - Highly interpretable results
    - Efficient with smaller datasets
    - Perfect for understanding basic patterns

    ### Gradient Boosting
    - Handles complex patterns
    - High prediction accuracy
    - Robust feature importance
    - Excellent for large datasets
    - Support for early stopping

    ### Support Vector Regression
    - Robust to outliers
    - Excellent generalization
    - Handles non-linear relationships
    - Kernel-based learning
    - Perfect for medium-sized datasets

    ### Model Comparison Capabilities
    - Automated performance metric calculation
    - Visual comparison tools
    - Feature importance analysis
    - Prediction accuracy comparison
    - Export functionality for detailed reports
    - Best model selection assistance
    - Cross-model validation

    ## Technical Details

    - Built with Streamlit for interactive web interface
    - Scikit-learn for machine learning models
    - Plotly for interactive visualizations
    - Pandas for efficient data manipulation
    - Advanced error handling and validation
    - Comprehensive model comparison framework
    - Robust data processing pipeline

    ## Data Security

    - Local data processing
    - No data storage without user consent
    - Secure model saving and loading
    - Privacy-focused design
    - Transparent data handling

    ## Support

    For support, feature requests, or bug reports, please contact:
    - Email: Bomino@mlawali.com
    - GitHub: [Project Repository](https://github.com/bomino/poultry)

    ## Version Information

    - Current Version: 2.0.0
    - Last Updated: November 2024
    - Key Features: 
      * Multi-model support with SVR
      * Advanced analytics and outlier detection
      * Comprehensive model comparison
      * Interactive visualizations
      * Exportable reports
      * Enhanced data validation
      * Improved error handling
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit by Bomino")

if __name__ == "__main__":
    app()