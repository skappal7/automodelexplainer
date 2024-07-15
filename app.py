import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from shapash import SmartExplainer

# Function to load model from UploadedFile
def load_model(file):
    file.seek(0)  # Reset file pointer to the beginning
    return pickle.load(file)

# Function to identify model type
def identify_model_type(model):
    if hasattr(model, 'predict_proba'):
        return 'classifier'
    elif hasattr(model, 'predict'):
        return 'regressor'
    else:
        return 'unknown'

# Function to run SHAP
def run_shap(model, X_test):
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"An error occurred while running SHAP: {e}")

# Function to run LIME
def run_lime(model, X_test, y_test):
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(X_test.values, mode='classification' if len(set(y_test)) > 2 else 'regression')
        i = np.random.randint(0, X_test.shape[0])
        exp = explainer.explain_instance(X_test.values[i], model.predict_proba if hasattr(model, 'predict_proba') else model.predict, num_features=10)
        exp.as_pyplot_figure()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"An error occurred while running LIME: {e}")

# Function to run Shapash
def run_shapash(model, X_test):
    try:
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_test)
        xpl.plot.features_importance()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"An error occurred while running Shapash: {e}")

# Streamlit app
st.title("Model Performance and Explainability Analyzer")

# File uploader for model and test data
model_file = st.file_uploader("Upload the .pkl model file", type=["pkl"])
test_data_file = st.file_uploader("Upload the test data file (CSV)", type=["csv"])

if model_file and test_data_file:
    model = load_model(model_file)
    test_data = pd.read_csv(test_data_file)
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    model_type = identify_model_type(model)
    st.write(f"Identified model type: {model_type}")

    # Allow user to select an explainer library
    explainer_choice = st.selectbox("Choose an explainer library", 
                                    ["SHAP", "LIME", "Shapash"])

    if st.button("Run Explainer"):
        if explainer_choice == "SHAP":
            run_shap(model, X_test)
        elif explainer_choice == "LIME":
            run_lime(model, X_test, y_test)
        elif explainer_choice == "Shapash":
            run_shapash(model, X_test)
        else:
            st.write("Please select a valid explainer library.")
