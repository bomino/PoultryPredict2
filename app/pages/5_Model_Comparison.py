import streamlit as st
import pandas as pd
from utils.model_comparison import ModelComparison
from models.model_factory import ModelFactory

def app():
    st.title("📊 Model Comparison")
    
    # Check if we have trained models in session state
    if 'trained_models' not in st.session_state:
        st.warning("No trained models available for comparison. Please train some models first!")
        return
        
    if len(st.session_state['trained_models']) < 2:
        st.warning("Please train at least two models to enable comparison!")
        return
    
    # Initialize model comparison
    comparison = ModelComparison()
    
    # Add results for each trained model
    for model_name, model_data in st.session_state['trained_models'].items():
        comparison.add_model_results(
            model_name=model_name,
            metrics=model_data['metrics'],
            predictions=model_data['predictions'],
            actual=model_data['actual'],
            feature_importance=model_data.get('feature_importance')
        )
    
    # Create tabs for different comparisons
    tab1, tab2, tab3, tab4 = st.tabs([
        "Metrics Comparison", 
        "Predictions Comparison", 
        "Feature Importance",
        "Model Rankings"
    ])
    
    with tab1:
        st.subheader("Model Metrics Comparison")
        metrics_df = comparison.get_metrics_comparison()
        st.dataframe(metrics_df)
        
        # Metric visualization
        metric_to_plot = st.selectbox(
            "Select metric to visualize",
            list(metrics_df.index),
            format_func=lambda x: x.upper()
        )
        
        metrics_plot = comparison.plot_metrics_comparison(metric_to_plot)
        st.plotly_chart(metrics_plot, use_container_width=True)
        
        # Show best model for each metric
        st.subheader("Best Models by Metric")
        for metric in metrics_df.index:
            best_model = comparison.get_best_model(metric)
            best_score = metrics_df.loc[metric, best_model]
            st.metric(
                label=f"Best {metric.upper()} Score", 
                value=f"{best_score:.4f}",
                help=f"Achieved by {best_model}"
            )
    
    with tab2:
        st.subheader("Predictions Comparison")
        
        # Plot predictions comparison
        pred_plot = comparison.plot_prediction_comparison()
        st.plotly_chart(pred_plot, use_container_width=True)
        
        # Show detailed predictions
        if st.checkbox("Show Detailed Predictions"):
            pred_df = comparison.get_prediction_comparison()
            st.dataframe(pred_df)
            
            # Download predictions
            st.download_button(
                label="Download Predictions",
                data=pred_df.to_csv(index=False),
                file_name="model_predictions_comparison.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.subheader("Feature Importance Comparison")
        
        importance_plot = comparison.plot_feature_importance_comparison()
        if importance_plot is not None:
            st.plotly_chart(importance_plot, use_container_width=True)
        else:
            st.info("Feature importance comparison not available for the selected models.")
    
    with tab4:
        st.subheader("Model Rankings")
        
        # Select metric for ranking
        ranking_metric = st.selectbox(
            "Select metric for ranking",
            list(metrics_df.index),
            format_func=lambda x: x.upper(),
            key="ranking_metric"
        )
        
        # Show rankings
        rankings = comparison.get_model_rankings(ranking_metric)
        
        # Create ranking display
        for rank, (model, score) in enumerate(rankings.items(), 1):
            st.write(f"{rank}. **{model}**: {score:.4f}")
    
    # Export comparison report
    st.sidebar.subheader("Export Comparison")
    if st.sidebar.button("Generate Report"):
        report = comparison.export_comparison_report()
        
        # Save report to session state
        st.session_state['comparison_report'] = report
        
        # Convert report to Excel
        output = pd.ExcelWriter('model_comparison_report.xlsx', engine='xlsxwriter')
        
        # Write each component
        report['metrics_comparison'].to_excel(output, sheet_name='Metrics')
        report['prediction_comparison'].to_excel(output, sheet_name='Predictions')
        
        # Rankings
        rankings_df = pd.DataFrame(report['model_rankings'])
        rankings_df.to_excel(output, sheet_name='Rankings')
        
        output.close()
        
        # Offer download
        with open('model_comparison_report.xlsx', 'rb') as f:
            st.sidebar.download_button(
                label="Download Report",
                data=f,
                file_name="model_comparison_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    app()