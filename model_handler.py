import os
import pandas as pd
import numpy as np
import importlib.util
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error

class ModelHandler:
    """
    Class to handle loading and running ML models from uploaded code.
    """
    
    def __init__(self, df, target_column, categorical_features, numerical_features, model_file_path):
        """
        Initialize the model handler.
        
        Args:
            df: DataFrame containing the data
            target_column: Name of the target column
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            model_file_path: Path to the Python file containing the model code
        """
        self.df = df
        self.target_column = target_column
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.model_file_path = model_file_path
        self.model = None
        self.create_model_func = self._load_model_function()
        self.train_data = None
        self.test_data = None
        self.preprocessor = None
        self.wrapper_model = None  # Wrapper model for DiCE
        
    def _load_model_function(self):
        """
        Load the create_model function from the uploaded Python file.
        
        Returns:
            function: The create_model function
        """
        try:
            # Load the module from the file path
            spec = importlib.util.spec_from_file_location("model_module", self.model_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the create_model function
            if hasattr(module, 'create_model'):
                return module.create_model
            else:
                raise ImportError("The uploaded file does not contain a 'create_model' function.")
        except Exception as e:
            raise ImportError(f"Error loading model function: {str(e)}")
    
    def train_model(self, selected_features=None):
        """
        Train the model using the uploaded data and model code.
        
        Args:
            selected_features: List of features to use for training. If None, all features will be used.
        """
        # Use all features if no specific features are selected
        if selected_features is None or not selected_features:
            selected_features = self.categorical_features + self.numerical_features
        
        # Store the selected features for later use
        self.selected_features = selected_features
        
        # Filter categorical and numerical features based on selection
        cat_features = [f for f in self.categorical_features if f in selected_features]
        num_features = [f for f in self.numerical_features if f in selected_features]
        
        # Split the data, using only selected features
        X = self.df[selected_features]
        y = self.df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save the test data for evaluation
        self.train_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
        
        # Create preprocessor
        transformers = []
        
        # Only add transformers for feature types that we have
        if num_features:
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, num_features))
            
        if cat_features:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, cat_features))
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
        # Fit the preprocessor on the training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Train the model
        self.model = self.create_model_func(X_train_processed, y_train)
        
        # Create wrapper model for DiCE that applies preprocessing before prediction
        self.wrapper_model = self.WrapperModel(self.model, self.preprocessor, self.selected_features, self.categorical_features)
        
    def predict(self, input_df):
        """
        Make a prediction using the trained model.
        
        Args:
            input_df: DataFrame containing the input data with raw categorical values
            
        Returns:
            array: Model predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Ensure we have all the selected features that were used during training
        if hasattr(self, 'selected_features'):
            # Only use the features that were selected for training
            input_filtered = input_df[self.selected_features]
        else:
            # Backward compatibility with older models
            input_filtered = input_df
        
        # Convert categorical columns to string type to ensure consistency and strip whitespace
        for cat_col in self.categorical_features:
            if cat_col in input_filtered.columns:
                input_filtered[cat_col] = input_filtered[cat_col].astype(str).str.strip()
        
        # Process the input data
        input_processed = self.preprocessor.transform(input_filtered)
        
        # Make the prediction
        return self.model.predict(input_processed)
    
    class WrapperModel:
        """
        Wrapper model to apply preprocessing before prediction.
        This is used to provide a model interface compatible with DiCE.
        """
        def __init__(self, model, preprocessor, selected_features, categorical_features):
            self.model = model
            self.preprocessor = preprocessor
            self.selected_features = selected_features
            self.categorical_features = categorical_features
        
        def predict(self, input_data):
            # If input is a DataFrame, apply preprocessing
            if hasattr(input_data, 'columns'):
                input_filtered = input_data[self.selected_features]
                
                # Convert categorical columns to string and strip whitespace
                for cat_col in self.categorical_features:
                    if cat_col in input_filtered.columns:
                        input_filtered[cat_col] = input_filtered[cat_col].astype(str).str.strip()
                
                # Preprocess
                input_processed = self.preprocessor.transform(input_filtered)
                
                # Predict
                return self.model.predict(input_processed)
            else:
                # Assume input is already preprocessed numpy array
                return self.model.predict(input_data)
        
        def predict_proba(self, input_df):
            if not hasattr(self.model, 'predict_proba'):
                raise AttributeError("Underlying model does not support predict_proba")
            
            # Select features
            input_filtered = input_df[self.selected_features]
            
            # Convert categorical columns to string and strip whitespace
            for cat_col in self.categorical_features:
                if cat_col in input_filtered.columns:
                    input_filtered[cat_col] = input_filtered[cat_col].astype(str).str.strip()
            
            # Preprocess
            input_processed = self.preprocessor.transform(input_filtered)
            
            # Predict probabilities
            return self.model.predict_proba(input_processed)
    
    def predict_proba(self, input_df):
        """
        Get prediction probabilities for classification models.
        
        Args:
            input_df: DataFrame containing the input data
            
        Returns:
            array: Prediction probabilities or None if not a classifier
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Ensure we have all the selected features that were used during training
        if hasattr(self, 'selected_features'):
            # Only use the features that were selected for training
            input_filtered = input_df[self.selected_features]
        else:
            # Backward compatibility with older models
            input_filtered = input_df
        
        # Process the input data
        input_processed = self.preprocessor.transform(input_filtered)
        
        # Check if model has predict_proba method (classification models)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(input_processed)
        else:
            return None
    
    def is_classification(self):
        """
        Check if the model is a classifier.
        
        Returns:
            bool: True if classifier, False if regressor
        """
        # Check if target is categorical or numerical
        if self.df[self.target_column].dtype == 'object' or self.df[self.target_column].nunique() < 10:
            return True
        else:
            return False
    
    def get_model_metrics(self):
        """
        Get metrics for the trained model.
        
        Returns:
            dict: Model metrics
        """
        if self.model is None or self.test_data is None:
            raise ValueError("Model has not been trained yet.")
        
        X_test, y_test = self.test_data
        X_test_processed = self.preprocessor.transform(X_test)
        y_pred = self.model.predict(X_test_processed)
        
        if self.is_classification():
            metrics = {
                'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                'Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                'F1 Score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4)
            }
        else:
            metrics = {
                'R² Score': round(r2_score(y_test, y_pred), 4),
                'Mean Squared Error': round(mean_squared_error(y_test, y_pred), 4),
                'Mean Absolute Error': round(mean_absolute_error(y_test, y_pred), 4)
            }
        
        return metrics
    
    def get_classes(self):
        """
        Get the classes for a classification model.
        
        Returns:
            array: Class labels or None if not a classifier
        """
        if not self.is_classification():
            return None
        
        # Get unique classes
        return self.df[self.target_column].unique()