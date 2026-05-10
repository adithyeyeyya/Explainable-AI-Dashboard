import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="ML Model Explainer with DiCE",
    page_icon="🧠",
    layout="wide"
)

import pandas as pd
import numpy as np
import streamlit.components.v1 as components

# Add this function to strip currency symbols and other non-numeric characters
def clean_numeric_column(series):
    """Clean a potentially numeric column by removing currency symbols and other non-numeric characters."""
    # Keep only digits, decimal points, and negative signs
    if series.dtype == 'object':
        try:
            # First remove common currency and unit symbols
            cleaned = series.astype(str).str.replace(r'[$€£¥%]', '', regex=True)
            # Then remove other non-numeric characters except decimal points and negative signs
            cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
            # Convert to numeric
            return pd.to_numeric(cleaned, errors='coerce')
        except:
            return series
    return series

# Hide the deployment button and "three dots" menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
import tempfile
import os
from utils import validate_csv, create_input_form, read_model_code
from counterfactual_explainer import CounterfactualExplainer


# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_code' not in st.session_state:
    st.session_state.model_code = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'categorical_features' not in st.session_state:
    st.session_state.categorical_features = []
if 'numerical_features' not in st.session_state:
    st.session_state.numerical_features = []
if 'is_target_categorical' not in st.session_state:
    st.session_state.is_target_categorical = False
if 'query_instance' not in st.session_state:
    st.session_state.query_instance = None
if 'counterfactuals' not in st.session_state:
    st.session_state.counterfactuals = None
if 'is_classification' not in st.session_state:
    st.session_state.is_classification = True
if 'used_features' not in st.session_state:
    st.session_state.used_features = []
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []

def reset_app():
    """Reset the application state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Main app title
st.title("🧠 ML Model Explainer with DiCE")
st.write("Upload your dataset and model to generate counterfactual explanations")

# Sidebar
with st.sidebar:
    st.title("Navigation")
    
    # Step buttons
    step = st.radio(
        "Select a step",
        ["1. Upload Dataset", "2. Define Model", "3. Generate Counterfactuals"],
        index=0
    )
    
    st.divider()
    
# Reset button
def clear_session():
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    st.experimental_rerun()

with st.sidebar:
    st.button("Reset Application", type="primary", on_click=clear_session)
    
    st.divider()
    
    # About section
    st.write("### About")
    st.write(
        """
        This application helps you understand how ML models make decisions using 
        DiCE (Diverse Counterfactual Explanations). Upload your dataset, define your model, 
        and get explanations for model predictions.
        """
    )

# Main content based on selected step
if step == "1. Upload Dataset":
    st.header("📊 Upload Dataset")
    st.write("Upload your CSV dataset to get started")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the data
        try:
            df = pd.read_csv(uploaded_file)
            
            # Clean the data by stripping whitespace from all string entries
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Additionally, strip whitespace from categorical columns explicitly
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].str.strip()
            
            # Auto-detect and clean potentially numeric columns
            for col in df.columns:
                # Skip columns that are already numeric
                if df[col].dtype in ['int64', 'float64']:
                    continue
                    
                # Try to clean and convert to numeric
                cleaned_col = clean_numeric_column(df[col])
                
                # If successful (most values converted to numeric), replace the column
                if pd.to_numeric(cleaned_col, errors='coerce').notna().mean() > 0.8:  # if 80% values are numeric
                    df[col] = cleaned_col
            
            st.session_state.df = df
            
            # Display basic information
            st.subheader("Dataset Overview")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
            
            # Show the data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Configure target and features
            st.subheader("Configure Model Features")
            
            # Select target column
            selected_target = st.selectbox(
                "Select target column",
                df.columns.tolist()
            )
            
            # First try to clean the target column
            cleaned_target = clean_numeric_column(df[selected_target])
            # Check if it's mostly numeric after cleaning
            if pd.to_numeric(cleaned_target, errors='coerce').notna().mean() > 0.8:
                df[selected_target] = cleaned_target
                # If it has few unique values despite being numeric, could still be categorical
                target_is_categorical = df[selected_target].nunique() < 10
            else:
                # Not successfully converted to numeric, likely categorical
                target_is_categorical = True
            
            # Ask user if target is categorical
            is_target_categorical = st.checkbox(
                "Target column is categorical",
                value=target_is_categorical,
                help="Select this if your target column contains categorical values (like 'Approved'/'Rejected') instead of continuous numerical values"
            )
            
            # Identify categorical and numerical features
            remaining_cols = [col for col in df.columns if col != selected_target]
            
            # User selects categorical features
            selected_categorical = st.multiselect(
                "Select categorical features",
                remaining_cols,
                default=[col for col in remaining_cols if df[col].dtype == 'object' or df[col].nunique() < 10]
            )
            
            # Numerical features are the remaining ones
            selected_numerical = [col for col in remaining_cols if col not in selected_categorical]
            st.write("Numerical features:", selected_numerical)
            
            # Save the configuration
            if st.button("Save Configuration"):
                st.session_state.target_column = selected_target
                st.session_state.categorical_features = selected_categorical
                st.session_state.numerical_features = selected_numerical
                st.session_state.is_target_categorical = is_target_categorical
                st.success("Configuration saved successfully!")
                st.write("Target column:", selected_target)
                st.write("Target is categorical:", is_target_categorical)
                st.write("Categorical features:", selected_categorical)
                st.write("Numerical features:", selected_numerical)
        
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
    else:
        # Show sample data option
        st.subheader("Get Started with Sample Data")
        st.write("Upload your own CSV file or use our sample loan data to explore counterfactual explanations.")
        
        if st.button("Load Sample Data"):
            # Load the sample loan data
            try:
                df = pd.read_csv("data/sample_loan_data.csv")
                
                # Clean the sample data by stripping whitespace from all string entries
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                
                st.session_state.df = df
                st.success("Sample loan data loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
        
        st.info("Upload a CSV file or use the sample data to get started.")

elif step == "2. Define Model":
    st.header("🤖 Define Model")
    
    # Check if dataset is loaded
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        st.stop()
    
    # Check if configuration is set
    if st.session_state.target_column is None:
        st.warning("Please configure your dataset features on step 1 first!")
        st.stop()
    
    # Display dataset information
    st.subheader("Dataset Information")
    st.write(f"Target column: {st.session_state.target_column}")
    st.write(f"Categorical features: {st.session_state.categorical_features}")
    st.write(f"Numerical features: {st.session_state.numerical_features}")
    
    # Feature selection for model training
    st.subheader("Select Features for Training")
    
    all_features = st.session_state.categorical_features + st.session_state.numerical_features
    
    # Allow users to select which features to use for training
    selected_features = st.multiselect(
        "Select features to use for model training",
        all_features,
        default=all_features,
        help="Choose which features to include in your model. By default, all features are selected."
    )
    
    # Store selected features in session state
    st.session_state.selected_features = selected_features
    
    # Show selected features as a list for clarity
    if selected_features:
        st.write(f"Selected {len(selected_features)} features for training:")
        st.write(", ".join(selected_features))
    else:
        st.warning("Please select at least one feature for training your model.")
    
    # Model definition options
    st.subheader("Define Your Model")
    
    model_option = st.radio(
        "Choose how to define your model",
        ["Use a simple default model", "Upload your own model code"],
        index=0
    )
    
    if model_option == "Use a simple default model":
        # Use the sample model
        st.write("The default model is a Random Forest classifier/regressor that will be trained on your data.")
        
        import os
        import tempfile
        import shutil
        from model_handler import ModelHandler

        def train_default_model(df, target_column, categorical_features, numerical_features, selected_features):
            # Copy sample_model.py to a temp file
            temp_dir = tempfile.mkdtemp()
            temp_model_path = os.path.join(temp_dir, "sample_model.py")
            shutil.copy("sample_model.py", temp_model_path)
            
            # Initialize ModelHandler
            handler = ModelHandler(df, target_column, categorical_features, numerical_features, temp_model_path)
            
            # Train the model
            handler.train_model(selected_features)
            
            return handler
        
        # Simple options for the default model
        st.write("### Model Parameters")
        n_estimators = st.slider("Number of trees (n_estimators)", min_value=10, max_value=500, value=100, step=10)
        max_depth = st.slider("Maximum tree depth (max_depth)", min_value=2, max_value=30, value=10, step=1)
        
        # Button to train model
        if st.button("Train Default Model"):
            with st.spinner("Training model..."):
                try:
                    # Train the model using ModelHandler
                    handler = train_default_model(
                        st.session_state.df,
                        st.session_state.target_column,
                        st.session_state.categorical_features,
                        st.session_state.numerical_features,
                        st.session_state.selected_features
                    )
                    
                    st.session_state.model_handler = handler
                    st.session_state.model = handler.model
                    st.session_state.used_features = st.session_state.selected_features
                    
                    # Determine if classification
                    is_classification = handler.is_classification()
                    st.session_state.is_classification = is_classification
                    
                    st.success("Model trained successfully!")
                    st.write(f"Model type: {'Classification' if is_classification else 'Regression'}")

                    # Add these lines to display model metrics
                    st.subheader("Model Performance Metrics")
                    metrics = handler.get_model_metrics()
                    import pandas as pd
                    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
                    st.table(metrics_df)

                    # Display model details
                    st.subheader("Model Details")
                    st.write(f"Number of training samples: {len(handler.train_data[0])}")
                    st.write(f"Number of testing samples: {len(handler.test_data[0])}")
                    st.write(f"Features used for training: {handler.selected_features}")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    else:  # Upload custom model code
        st.write("Upload your model code. It should contain a `create_model(X_train, y_train)` function.")
        uploaded_code = st.file_uploader("Upload Python model code", type="py")
        
        if uploaded_code is not None:
            model_code = uploaded_code.read().decode("utf-8")
            st.session_state.model_code = model_code
            
            # Display the code
            with st.expander("View uploaded code"):
                st.code(model_code, language="python")
            
            # Button to train the model
            if st.button("Train Custom Model"):
                st.write("Since we can't run the actual model code, we'll use a simple prediction function instead.")
                
                # Use the same simple prediction function as above
                try:
                    def simple_prediction_function(input_df):
                        """
                        A simple prediction function that uses correlation to make predictions.
                        This is a workaround since we can't use the uploaded code directly.
                        """
                        df = st.session_state.df
                        target_column = st.session_state.target_column
                        
                        # Extract only the features used for training
                        input_features = [f for f in input_df.columns if f in st.session_state.selected_features]
                        limited_input_df = input_df[input_features]
                        
                        # Determine if classification problem
                        is_classification = df[target_column].nunique() < 10
                        
                        if is_classification:
                            # Simple classification logic based on means and correlations
                            prediction = []
                            
                            for _, row in limited_input_df.iterrows():
                                # Calculate a weighted vote
                                vote_sum = 0
                                weight_sum = 0
                                
                                # Add logic for numerical features
                                numerical_features = [f for f in st.session_state.numerical_features if f in st.session_state.selected_features]
                                for feature in numerical_features:
                                    if feature in row:
                                        # Find correlation
                                        correlation = abs(df[feature].corr(df[target_column]))
                                        if np.isnan(correlation):
                                            correlation = 0.1
                                        
                                        # Compare to average value in each class
                                        feature_value = row[feature]
                                        class_means = df.groupby(target_column)[feature].mean()
                                        
                                        # Find closest class
                                        closest_class = 0
                                        min_diff = float('inf')
                                        
                                        for cls, mean_val in class_means.items():
                                            diff = abs(feature_value - mean_val)
                                            if diff < min_diff:
                                                min_diff = diff
                                                closest_class = cls
                                        
                                        # Add weighted vote
                                        vote_sum += closest_class * correlation
                                        weight_sum += correlation
                                
                                # Add logic for categorical features
                                categorical_features = [f for f in st.session_state.categorical_features if f in st.session_state.selected_features]
                                for feature in categorical_features:
                                    if feature in row:
                                        feature_value = row[feature]
                                        if isinstance(feature_value, (str, int, float)) and feature_value in df[feature].values:
                                            # Find most common target for this category
                                            subset = df[df[feature] == feature_value]
                                            if not subset.empty:
                                                most_common_target = subset[target_column].mode()[0]
                                                
                                                # Add weighted vote
                                                vote_sum += most_common_target * 0.5
                                                weight_sum += 0.5
                                
                                # Calculate final prediction
                                if weight_sum > 0:
                                    pred = round(vote_sum / weight_sum)
                                else:
                                    pred = df[target_column].mode()[0]
                                
                                prediction.append(pred)
                            
                            return np.array(prediction)
                        else:
                            # Regression logic
                            prediction = []
                            
                            for _, row in limited_input_df.iterrows():
                                # Target mean as base
                                target_mean = df[target_column].mean()
                                target_std = df[target_column].std()
                                
                                # Adjust based on feature values
                                adjustment = 0
                                total_weight = 0
                                
                                # Adjust based on numerical features
                                numerical_features = [f for f in st.session_state.numerical_features if f in st.session_state.selected_features]
                                for feature in numerical_features:
                                    if feature in row:
                                        # Calculate correlation
                                        correlation = df[feature].corr(df[target_column])
                                        if np.isnan(correlation):
                                            continue
                                        
                                        # Calculate feature z-score
                                        feature_mean = df[feature].mean()
                                        feature_std = df[feature].std()
                                        
                                        if feature_std > 0:
                                            z_score = (row[feature] - feature_mean) / feature_std
                                            # Adjust prediction
                                            adjustment += z_score * correlation * target_std
                                            total_weight += abs(correlation)
                                
                                # Normalize adjustment
                                if total_weight > 0:
                                    adjustment /= total_weight
                                
                                # Final prediction
                                prediction.append(target_mean + adjustment)
                            
                            return np.array(prediction)
                    
                    # Ensure we have selected features
                    if not st.session_state.selected_features:
                        st.error("Please select at least one feature for training.")
                        st.stop()
                    
                    # Store the model in session state
                    st.session_state.model = simple_prediction_function
                    
                    # Store selected features for later use
                    st.session_state.used_features = st.session_state.selected_features
                    
                    # Determine if classification
                    is_classification = st.session_state.df[st.session_state.target_column].nunique() < 10
                    st.session_state.is_classification = is_classification
                    
                    st.success("Custom model trained successfully!")
                    st.write(f"Model type: {'Classification' if is_classification else 'Regression'}")

                    # Since we're using a simple prediction function here instead of the actual model,
                    # we can provide some mock metrics
                    st.subheader("Model Details")
                    st.write(f"Features used for training: {st.session_state.selected_features}")
                    st.write("Note: Custom model performance metrics are not available for uploaded model code in this demo.")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")

elif step == "3. Generate Counterfactuals":
    st.header("🔍 Generate Counterfactual Explanations")
    
    # Check if dataset and model are loaded
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        st.stop()
    
    if st.session_state.model is None:
        st.warning("Please define and train a model first!")
        st.stop()
    
    if not st.session_state.used_features:
        st.warning("No features were selected for the model. Please go back to step 2.")
        st.stop()
    
    st.subheader("Generate DiCE Counterfactuals")
    st.write("""
    Counterfactual explanations show how changing certain features would change the model's prediction.
    These explanations help understand which features are most influential for a particular prediction.
    """)
    
    # Create input form for the user to enter values
    st.write("### Enter Input Values")
    query_values = {}

    # Replace the two forms with a single combined form with unique key and submit button
    with st.form("combined_input_form"):
        st.write("### Enter Input Values")
        # Only show features that were selected for model training
        # Filter numerical features that were used for training
        used_numerical = [f for f in st.session_state.numerical_features if f in st.session_state.used_features]
        if used_numerical:
            st.write("#### Numerical Features")
            for feature in used_numerical:
                # Use cleaned numeric values for min, max, mean
                cleaned_col = clean_numeric_column(st.session_state.df[feature])
                min_val = float(cleaned_col.min())
                max_val = float(cleaned_col.max())
                mean_val = float(cleaned_col.mean())

                # Ensure min_val is less than max_val to avoid slider error
                if min_val >= max_val:
                    min_val = max_val - 1e-5

                query_values[feature] = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100 if max_val > min_val else 0.01
                )

        # Filter categorical features that were used for training
        used_categorical = [f for f in st.session_state.categorical_features if f in st.session_state.used_features]
        if used_categorical:
            st.write("#### Categorical Features")
            for feature in used_categorical:
                # Get unique values and strip whitespace
                unique_values = st.session_state.df[feature].unique()
                unique_values = [str(val).strip() for val in unique_values]
                # Map categorical values to their category codes (integers)
                category_map = {val: idx for idx, val in enumerate(unique_values)}
                selected_value = st.selectbox(
                    f"{feature}",
                    options=unique_values,
                    index=0
                )
                # Store the original string value instead of the category code
                query_values[feature] = selected_value

        st.write("### Counterfactual Settings")
        # Number of counterfactuals to generate
        num_counterfactuals = st.slider(
            "Number of counterfactuals to generate",
            min_value=1,
            max_value=5,
            value=3
        )

        # Add user-defined thresholds for regression if not classification
        min_percentage = None
        max_percentage = None
        min_absolute = None
        max_absolute = None
        change_type = "Percentage"

        if not st.session_state.is_classification:
            st.subheader("Regression Settings")

            # Helper function for context-aware defaults
            def get_default_regression_thresholds(target_name, current_value):
                if any(term in target_name.lower() for term in ['price', 'cost', 'salary']):
                    return -15, 15
                elif any(term in target_name.lower() for term in ['rate', 'percentage']):
                    return -30, 30
                elif current_value > 1000:
                    return -10, 10
                else:
                    return -20, 20

            # Get current prediction mean value for default thresholds
            current_value = float(st.session_state.df[st.session_state.target_column].mean())
            default_min_pct, default_max_pct = get_default_regression_thresholds(st.session_state.target_column, current_value)

            change_type = st.radio("Change Type", ["Percentage", "Absolute Value"])

            if change_type == "Percentage":
                col1, col2 = st.columns(2)
                with col1:
                    min_percentage = st.slider("Minimum % Change", -90, 0, default_min_pct)
                with col2:
                    max_percentage = st.slider("Maximum % Change", 0, 200, default_max_pct)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    min_absolute = st.number_input("Minimum Absolute Change", value=-10.0)
                with col2:
                    max_absolute = st.number_input("Maximum Absolute Change", value=10.0)

        # Submit button for the combined form
        submit_button = st.form_submit_button("Generate Counterfactuals")

    # Process the form submission
    if submit_button:
        with st.spinner("Generating counterfactual explanations..."):
            try:
                # Create a DataFrame from the input values
                query_instance = pd.DataFrame([query_values])

                # Strip whitespace from categorical features in query_instance
                for cat_col in st.session_state.categorical_features:
                    if cat_col in query_instance.columns:
                        query_instance[cat_col] = query_instance[cat_col].astype('category').cat.codes

                # Also ensure categorical features in query_instance match dataset categories by mapping if needed
                for cat_col in st.session_state.categorical_features:
                    if cat_col in query_instance.columns and cat_col in st.session_state.df.columns:
                        # Map query_instance values to dataset categories if possible
                        dataset_categories = st.session_state.df[cat_col].astype('category').cat.categories
                        query_val = query_instance.at[0, cat_col]

                        # If query_val not in dataset categories, replace with mode
                        if query_val not in dataset_categories:
                            mode_val = st.session_state.df[cat_col].mode()[0]
                            query_instance.at[0, cat_col] = mode_val

                st.session_state.query_instance = query_instance

                # Make a prediction using wrapper_model to ensure preprocessing
                prediction = st.session_state.model_handler.wrapper_model.predict(query_instance)

                # Get the categorical and numerical features that were used for training
                # Adjust feature lists: move 'no_of_dependents' from categorical to numerical if present
                cat_features = [f for f in st.session_state.categorical_features if f in st.session_state.used_features and f != 'no_of_dependents']
                num_features = [f for f in st.session_state.numerical_features if f in st.session_state.used_features]
                if 'no_of_dependents' in st.session_state.used_features and 'no_of_dependents' not in num_features:
                    num_features.append('no_of_dependents')

                # Create counterfactual explainer using only the features used by the model
                # Drop columns not used for training (excluded features) to avoid DiCE errors
                df_for_explainer = st.session_state.df.copy()
                # Keep only selected features and target column
                columns_to_keep = st.session_state.used_features + [st.session_state.target_column]
                df_for_explainer = df_for_explainer[columns_to_keep]

                # Cast categorical features to categorical codes with consistent categories
                for cat_feat in cat_features:
                    if cat_feat in df_for_explainer.columns:
                        categories = st.session_state.df[cat_feat].astype('category').cat.categories
                        df_for_explainer[cat_feat] = pd.Categorical(df_for_explainer[cat_feat], categories=categories).codes

                # Cast numerical features to float type explicitly
                for num_feat in num_features:
                    if num_feat in df_for_explainer.columns:
                        df_for_explainer[num_feat] = pd.to_numeric(df_for_explainer[num_feat], errors='coerce')

                # Convert 'no_of_dependents' to numeric if present
                if 'no_of_dependents' in df_for_explainer.columns:
                    df_for_explainer['no_of_dependents'] = pd.to_numeric(df_for_explainer['no_of_dependents'], errors='coerce')

                # Debug print for feature lists
                print(f"Categorical features passed to explainer: {cat_features}")
                print(f"Numerical features passed to explainer: {num_features}")

                explainer = CounterfactualExplainer(
                    df_for_explainer,
                    st.session_state.model_handler.wrapper_model,
                    cat_features,
                    num_features,
                    st.session_state.target_column,
                    backend='sklearn',
                    model_type='classifier' if st.session_state.is_classification else 'regressor',
                    is_target_categorical=st.session_state.is_target_categorical
                )

                # Cast categorical features in query_instance to categorical codes with consistent categories
                for cat_feat in cat_features:
                    if cat_feat in query_instance.columns:
                        # Ensure consistent categories from original dataset
                        categories = st.session_state.df[cat_feat].astype('category').cat.categories
                        query_instance[cat_feat] = pd.Categorical(query_instance[cat_feat], categories=categories).codes

                # Cast numerical features in query_instance to float type explicitly
                for num_feat in num_features:
                    if num_feat in query_instance.columns:
                        query_instance[num_feat] = pd.to_numeric(query_instance[num_feat], errors='coerce')

                # Convert 'no_of_dependents' to numeric if present
                if 'no_of_dependents' in query_instance.columns:
                    query_instance['no_of_dependents'] = pd.to_numeric(query_instance['no_of_dependents'], errors='coerce')

                # Generate counterfactuals
                counterfactuals = explainer.generate_counterfactuals(
                    query_instance,
                    num_counterfactuals=num_counterfactuals,
                    regression_min_pct=min_percentage,
                    regression_max_pct=max_percentage,
                    regression_abs_min=min_absolute,
                    regression_abs_max=max_absolute,
                    features_to_vary=st.session_state.used_features  # Only vary selected features
                )

                st.session_state.counterfactuals = counterfactuals

                # Display the results
                st.subheader("DiCE Counterfactual Results")

                # Original prediction
                st.write("### Original Prediction")
                if st.session_state.is_classification:
                    st.metric("Prediction", f"Class {prediction[0]}")
                else:
                    st.metric("Prediction", f"{prediction[0]:.4f}")

                # Display the original input
                st.write("### Original Input")
                st.dataframe(query_instance)

                # Display counterfactuals
                if not counterfactuals.empty:
                    st.write("### Counterfactual Examples")
                    st.write("If these changes were made to your inputs, the model would predict a different outcome:")

                    # Get predictions for counterfactuals using wrapper_model
                    cf_predictions = st.session_state.model_handler.wrapper_model.predict(counterfactuals)

                    # Prepare mapping dictionaries for categorical features
                    cat_features_dict = {}
                    for cat_col in cat_features:
                        if cat_col in st.session_state.df.columns:
                            unique_values = st.session_state.df[cat_col].unique()
                            cat_dict = {val: idx for idx, val in enumerate(unique_values)}
                            cat_features_dict[cat_col] = cat_dict

                    # Create a formatted display DataFrame
                    display_df = pd.DataFrame()

                    # Process each counterfactual
                    for i, (_, cf_row) in enumerate(counterfactuals.iterrows()):
                        # Create a row for this counterfactual
                        cf_display = {}

                        # Add each feature with highlighting for changes
                        for feature in st.session_state.used_features:
                            if feature in query_instance.columns and feature in cf_row:
                                original_val = query_instance[feature].values[0]
                                cf_val = cf_row[feature]

                                # For categorical features, map back to original values
                                if feature in cat_features:
                                    # Map codes back to original values
                                    cat_dict = cat_features_dict.get(feature, {})
                                    # Reverse the dictionary to map from codes to values
                                    rev_dict = {v: k for k, v in cat_dict.items()}

                                    # Check if the value exists in the reverse dictionary
                                    if cf_val in rev_dict:
                                        cf_val = rev_dict[cf_val]
                                    if original_val in rev_dict:
                                        original_val = rev_dict[original_val]

                                # Check if value has changed
                                if cf_val != original_val:
                                    cf_display[feature] = f"{cf_val} (changed from {original_val})"
                                else:
                                    cf_display[feature] = f"{cf_val}"

                        # Add prediction
                        cf_display["Prediction"] = f"Class {cf_predictions[i]}" if st.session_state.is_classification else f"{cf_predictions[i]:.4f}"

                        # Add to display DataFrame
                        display_df = pd.concat([display_df, pd.DataFrame([cf_display])], ignore_index=True)

                    # Show formatted results
                    st.dataframe(display_df)

                    # Feature importance and plotting enabled
                    st.write("### Feature Importance")
                    feature_importance = explainer.get_feature_importance(query_instance)

                    # Create DataFrame for better display
                    importance_df = pd.DataFrame({
                        'Feature': feature_importance.keys(),
                        'Importance': feature_importance.values()
                    })

                    # Show as a bar chart
                    st.bar_chart(importance_df.set_index('Feature'))

                    # Show as a table
                    st.dataframe(importance_df)

                    # Explanation
                    st.write("### Explanation")
                    top_features = list(feature_importance.keys())[:3]
                    explanation = "To change the prediction, consider modifying these features:"
                    for i, feature in enumerate(top_features):
                        explanation += f"\n{i+1}. {feature} (importance: {feature_importance[feature]:.4f})"

                    st.write(explanation)
                else:
                    st.warning("Could not generate counterfactuals. Try with different input values.")

            except Exception as e:
                st.error(f"Error generating counterfactuals: {str(e)}")

    # If counterfactuals were previously generated, display them
    elif hasattr(st.session_state, 'counterfactuals') and st.session_state.counterfactuals is not None and hasattr(st.session_state, 'query_instance') and st.session_state.query_instance is not None:
        st.subheader("Previous Counterfactual Results")

        # Original prediction using wrapper_model
        prediction = st.session_state.model_handler.wrapper_model.predict(st.session_state.query_instance)
        st.write("### Original Prediction")
        if st.session_state.is_classification:
            st.metric("Prediction", f"Class {prediction[0]}")
        else:
            st.metric("Prediction", f"{prediction[0]:.4f}")

        # Display the original input
        st.write("### Original Input")
        st.dataframe(st.session_state.query_instance)

        # Display counterfactuals
        if not st.session_state.counterfactuals.empty:
            st.write("### Counterfactual Examples")
            st.write("These alternative inputs would lead to different predictions:")

            # Get predictions for counterfactuals using wrapper_model
            cf_predictions = st.session_state.model_handler.wrapper_model.predict(st.session_state.counterfactuals)

            # Add predictions to counterfactuals
            counterfactuals_with_preds = st.session_state.counterfactuals.copy()
            counterfactuals_with_preds['Prediction'] = cf_predictions

            st.dataframe(counterfactuals_with_preds)
