

import streamlit as st
from config.settings import APP_NAME, APP_ICON, LAYOUT

# Configure the Streamlit page
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout=LAYOUT
)

# Main page header
st.title("🐔 Poultry Weight Predictor")

# Welcome message
st.markdown("""
## Welcome to the Poultry Weight Predictor

This application helps you predict poultry weight based on environmental and feeding data. 
You can:

1. Upload and analyze your poultry data
2. Train machine learning models
3. Make predictions on new data
4. Visualize results and insights

### Getting Started

Use the sidebar to navigate through different sections of the app:

- **Data Upload**: Upload and preview your data
- **Data Analysis**: Explore your data with visualizations
- **Model Training**: Train and evaluate prediction models
- **Predictions**: Make predictions on new data

### Required Data Format

Your CSV file should include the following columns:
- Internal Temperature (Int Temp)
- Internal Humidity (Int Humidity)
- Air Temperature (Air Temp)
- Wind Speed
- Feed Intake
- Weight (target variable)
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")