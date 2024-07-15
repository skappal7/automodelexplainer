import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

def load_data(file):
    return pd.read_csv(file)

def evaluate_model(y_true, y_pred):
    try:
        if y_true.dtype == 'object' or y_true.dtype == 'bool':
            score = accuracy_score(y_true, y_pred)
            return f"Accuracy: {score:.4f}"
        else:
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return f"Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}"
    except Exception as e:
        return f"Error in evaluation: {str(e)}"

def explain_predictions(X, y_pred):
    explainer = shap.KernelExplainer(lambda x: x, X)
    shap_values = explainer.shap_values(X)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)

st.title("Model Performance Analyzer")

uploaded_data = st.file_uploader("Upload your test data with predictions (.csv)", type="csv")

if uploaded_data is not None:
    try:
        data = load_data(uploaded_data)
        
        st.write("Data Preview:")
        st.write(data.head())
        
        feature_cols = st.multiselect("Select feature columns", data.columns)
        target_col = st.selectbox("Select target column", data.columns)
        prediction_col = st.selectbox("Select prediction column", data.columns)
        
        if feature_cols and target_col and prediction_col:
            X = data[feature_cols]
            y_true = data[target_col]
            y_pred = data[prediction_col]
            
            st.write(f"Features: {feature_cols}")
            st.write(f"Target: {target_col}")
            st.write(f"Predictions: {prediction_col}")
            
            performance = evaluate_model(y_true, y_pred)
            st.write("Model Performance:", performance)
            
            if st.button("Explain Predictions"):
                explain_predictions(X, y_pred)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
else:
    st.write("Please upload a CSV file containing your test data and model predictions.")
