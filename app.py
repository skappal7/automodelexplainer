import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.base import is_classifier, is_regressor
import shap
import lime
import lime.lime_tabular
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def load_model(file):
    try:
        model_dict = pickle.load(file)
        
        if isinstance(model_dict, (RandomForestClassifier, RandomForestRegressor)):
            return model_dict
        
        if isinstance(model_dict, dict) and 'estimators_' in model_dict:
            if 'classes_' in model_dict:
                new_model = RandomForestClassifier()
            else:
                new_model = RandomForestRegressor()
            
            new_model.n_estimators = len(model_dict['estimators_'])
            new_model.estimators_ = model_dict['estimators_']
            if 'classes_' in model_dict:
                new_model.classes_ = model_dict['classes_']
            new_model.n_features_in_ = model_dict['n_features_in_']
            
            for param in ['n_outputs_', 'n_classes_', 'max_features_']:
                if param in model_dict:
                    setattr(new_model, param, model_dict[param])
            
            return new_model
        else:
            raise ValueError("Unrecognized model format")
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def load_data(file):
    return pd.read_csv(file)

def determine_model_type(model):
    if is_classifier(model):
        return "classifier"
    elif is_regressor(model):
        return "regressor"
    else:
        return "unknown"

def get_suitable_explainers(model_type):
    if model_type in ["classifier", "regressor"]:
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

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Model Explainer App")

uploaded_model = st.file_uploader("Choose a .pkl model file", type="pkl")
uploaded_data = st.file_uploader("Choose a CSV file with test data", type="csv")

if uploaded_model is not None and uploaded_data is not None:
    try:
        model = load_model(uploaded_model)
        st.write(f"Model type: {type(model).__name__}")
        
        data = load_data(uploaded_data)
        
        st.write("Data Preview:")
        st.write(data.head())
        
        model_type = determine_model_type(model)
        st.write(f"Detected model type: {model_type}")

        suitable_explainers = get_suitable_explainers(model_type)
        
        if suitable_explainers:
            explainer_choice = st.selectbox("Choose an explainer", suitable_explainers)
            
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            st.write(f"Features: {X.columns.tolist()}")
            st.write(f"Target: {y.name}")
            
            if st.button("Explain Model"):
                explain_model(model, X, explainer_choice)
        else:
            st.write("No suitable explainers found for this model type.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
else:
    st.write("Please upload both a model file (.pkl) and a test data file (.csv) to proceed.")
