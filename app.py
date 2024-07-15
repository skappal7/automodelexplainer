import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import shap

def load_model(file):
    return pickle.load(file)

def load_data(file):
    return pd.read_csv(file)

def evaluate_model(model, X, y):
    try:
        y_pred = model.predict(X)
        if model._estimator_type == "classifier":
            score = accuracy_score(y, y_pred)
            return f"Accuracy: {score:.4f}"
        else:
            score = mean_squared_error(y, y_pred, squared=False)
            return f"Root Mean Squared Error: {score:.4f}"
    except Exception as e:
        return f"Error in evaluation: {str(e)}"

def explain_model(model, X):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        st.pyplot(shap.summary_plot(shap_values, X, plot_type="bar"))
    except Exception as e:
        st.write(f"Error in SHAP explanation: {str(e)}")

st.title("Model Performance Analyzer")

uploaded_model = st.file_uploader("Upload your pickled model (.pkl)", type="pkl")
uploaded_data = st.file_uploader("Upload your test data (.csv)", type="csv")

if uploaded_model is not None and uploaded_data is not None:
    try:
        model = load_model(uploaded_model)
        data = load_data(uploaded_data)
        
        st.write("Data Preview:")
        st.write(data.head())
        
        st.write(f"Model type: {type(model).__name__}")
        
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        st.write(f"Features: {X.columns.tolist()}")
        st.write(f"Target: {y.name}")
        
        performance = evaluate_model(model, X, y)
        st.write("Model Performance:", performance)
        
        if st.button("Explain Model"):
            explain_model(model, X)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
else:
    st.write("Please upload both a model file (.pkl) and a test data file (.csv) to proceed.")
