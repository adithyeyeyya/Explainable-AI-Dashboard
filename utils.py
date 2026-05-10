import streamlit as st
import pandas as pd
import numpy as np
import io

def validate_csv(uploaded_file):
    """
    Validate and load a CSV file.
    
    Args:
        uploaded_file: The uploaded CSV file
        
    Returns:
        tuple: (DataFrame, error_message)
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Basic validation
        if df.empty:
            return None, "The uploaded file is empty."
        
        if df.shape[1] < 2:
            return None, "The dataset must have at least 2 columns (features and target)."
        
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            return df, f"Warning: Dataset contains {missing_count} missing values. Results may be affected."
        
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def create_input_form(df, categorical_features, numerical_features):
    """
    Create an input form based on dataset features.
    
    Args:
        df: DataFrame containing the data
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        
    Returns:
        dict: Form values
    """
    form_values = {}
    
    with st.form("prediction_form"):
        st.write("Enter values for each feature:")
        
        # Create form fields for numerical features
        for feature in numerical_features:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            
            form_values[feature] = st.slider(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100
            )
        
        # Create form fields for categorical features
        for feature in categorical_features:
            unique_values = df[feature].unique().tolist()
            form_values[feature] = st.selectbox(
                f"{feature}",
                options=unique_values,
                index=0
            )
        
        # Submit button
        submit_button = st.form_submit_button("Make Prediction")
        
        if submit_button:
            return form_values
    
    return None

def read_model_code(file_content):
    """
    Read and validate the model code.
    
    Args:
        file_content: The content of the uploaded Python file
        
    Returns:
        str: The model code as a string
    """
    try:
        # Check if the code contains the required function
        if "def create_model" not in file_content:
            st.error("The uploaded code must contain a 'create_model(X_train, y_train)' function.")
            return None
        
        return file_content
    except Exception as e:
        st.error(f"Error reading model code: {str(e)}")
        return None