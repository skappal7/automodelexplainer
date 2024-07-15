import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.base import is_classifier, is_regressor
import shap
import lime
import lime.lime_tabular
from shapash.explainer.smart_explainer import SmartExplainer

def load_model(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def determine_model_type(model):
    if is_classifier(model):
        return "classifier"
    elif is_regressor(model):
        return "regressor"
    else:
        return "unknown"

def get_suitable_explainers(model_type):
    if model_type == "classifier":
        return ["SHAP", "LIME", "Shapash"]
    elif model_type == "regressor":
        return ["SHAP", "LIME", "Shapash"]
    else:
        return []

def explain_model(model, X, explainer_choice):
    if explainer_choice == "SHAP":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        st.pyplot(shap.summary_plot(shap_values, X))
    elif explainer_choice == "LIME":
        explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns)
        exp = explainer.explain_instance(X.iloc[0], model.predict_proba)
        st.write(exp.as_list())
    elif explainer_choice == "Shapash":
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X)
        st.write(xpl.to_pandas())

st.title("Model Explainer App")

uploaded_file = st.file_uploader("Choose a .pkl model file", type="pkl")

if uploaded_file is not None:
    model = load_model(uploaded_file)
    model_type = determine_model_type(model)
    st.write(f"Detected model type: {model_type}")

    suitable_explainers = get_suitable_explainers(model_type)
    
    if suitable_explainers:
        explainer_choice = st.selectbox("Choose an explainer", suitable_explainers)
        
        # For demonstration, we're using a dummy dataset. 
        # In a real scenario, you'd need to provide the actual data used to train the model.
        X = pd.DataFrame(np.random.rand(100, 5), columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        
        if st.button("Explain Model"):
            explain_model(model, X, explainer_choice)
    else:
        st.write("No suitable explainers found for this model type.")
