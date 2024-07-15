import streamlit as st
import pickle
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from shapash import SmartExplainer

# Function to load model
def load_model(file):
    with open(file, 'rb') as f:
        model = pickle.load(f)
    return model

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
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)
    st.pyplot(plt)

# Function to run LIME
def run_lime(model, X_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_test.values)
    i = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(X_test.values[i], model.predict_proba, num_features=10)
    exp.as_pyplot_figure()
    st.pyplot(plt)

# Function to run Shapash
def run_shapash(model, X_test):
    xpl = SmartExplainer(model=model)
    xpl.compile(x=X_test)
    xpl.plot.features_importance()
    st.pyplot(plt)

# Function to run Explainable AI (hypothetical)
def run_explainable_ai(model, X_test):
    explainable_ai_function(model, X_test)
    st.pyplot(plt)

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
                                    ["SHAP", "LIME", "Shapash", "Explainable AI (hypothetical)"])

    if st.button("Run Explainer"):
        if explainer_choice == "SHAP":
            run_shap(model, X_test)
        elif explainer_choice == "LIME":
            run_lime(model, X_test)
        elif explainer_choice == "Shapash":
            run_shapash(model, X_test)
        elif explainer_choice == "Explainable AI (hypothetical)":
            run_explainable_ai(model, X_test)
        else:
            st.write("Please select a valid explainer library.")
