import pandas as pd
import numpy as np

class ModelHandlerImproved:
    def __init__(self, model, features, target_column=None, categoricals=None, numericals=None):
        """
        Initialize the model handler with clear separation between features and target.
        
        Args:
            model: The trained ML model (must implement predict or predict_proba)
            features: List of feature names used by the model
            target_column: Name of the target column (what the model predicts)
            categoricals: List of categorical feature names
            numericals: List of numerical feature names
        """
        self.model = model
        self.target_column = target_column
        
        # Store selected features, ensuring target is not included if specified
        if target_column and target_column in features:
            self.selected_features = [f for f in features if f != target_column]
            print(f"Notice: Removed target column '{target_column}' from feature list")
        else:
            self.selected_features = features.copy()
        
        # Track feature types for easier processing
        self.categorical_features = categoricals if categoricals else []
        self.numerical_features = numericals if numericals else []
        
        # Verify feature type lists match selected features
        unknown_cats = [f for f in self.categorical_features if f not in self.selected_features]
        unknown_nums = [f for f in self.numerical_features if f not in self.selected_features]
        
        if unknown_cats:
            print(f"Warning: Categorical features not in selected features: {unknown_cats}")
        if unknown_nums:
            print(f"Warning: Numerical features not in selected features: {unknown_nums}")
        
        # Ensure all features are categorized
        uncategorized = set(self.selected_features) - set(self.categorical_features) - set(self.numerical_features)
        if uncategorized:
            print(f"Warning: These features have no type specified: {uncategorized}")

    def predict(self, input_data):
        """
        Make predictions using the trained model on new input data.
        Handles cases where input data might not match the exact features used during training.
        
        Args:
            input_data: DataFrame containing features for prediction
            
        Returns:
            Predictions from the model
        """
        # Deep copy to avoid modifying the original data
        input_copy = input_data.copy()
        
        # Get features actually needed by the model (exclude target if present)
        required_features = [f for f in self.selected_features if f != self.target_column]
        
        # Check for missing features
        missing_features = set(required_features) - set(input_copy.columns)
        if missing_features:
            error_msg = f"Missing required features for prediction: {missing_features}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Check for extra features that aren't needed
        extra_features = set(input_copy.columns) - set(required_features)
        if extra_features and self.target_column in extra_features:
            # If target column is in input but shouldn't be used for prediction, remove it
            print(f"Notice: Removing target column '{self.target_column}' from prediction inputs")
            if self.target_column in input_copy.columns:
                input_copy = input_copy.drop(columns=[self.target_column])
            extra_features.remove(self.target_column)
        
        if extra_features:
            print(f"Notice: Input contains extra features that won't be used: {extra_features}")
        
        # Filter to only use required features in the correct order
        available_required_features = [f for f in required_features if f in input_copy.columns]
        input_filtered = input_copy[available_required_features]
        
        # Verify we have all features needed
        if len(available_required_features) != len(required_features):
            missing = set(required_features) - set(available_required_features)
            raise ValueError(f"Cannot make prediction: missing features {missing}")
        
        try:
            # Make prediction
            result = self.model.predict(input_filtered)
            return result
        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")

    def setup_counterfactual_explainer(self, dataframe, wrapper_model):
        """
        Set up the counterfactual explainer with proper feature/target separation.
        
        Args:
            dataframe: DataFrame containing both features and target
            wrapper_model: Model wrapper implementing predict/predict_proba
            
        Returns:
            Initialized CounterfactualExplainer object
        """
        try:
            # Ensure target column is in the dataframe
            if self.target_column not in dataframe.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in dataframe for counterfactual analysis")
            
            # Print categorical and numerical features for debugging/validation
            print(f"Categorical features passed to explainer: {self.categorical_features}")
            print(f"Numerical features passed to explainer: {self.numerical_features}")
            
            # Initialize the counterfactual explainer
            from counterfactual_explainer import CounterfactualExplainer
            
            cf_explainer = CounterfactualExplainer(
                df=dataframe,
                model=wrapper_model,  # Use wrapper model that implements predict/predict_proba
                categorical_features=self.categorical_features,
                numerical_features=self.numerical_features,
                target_column=self.target_column
            )
            
            return cf_explainer
        except Exception as e:
            print(f"Error setting up counterfactual explainer: {str(e)}")
            return None

class ModelWrapper:
    """
    Wrapper class to standardize model interface for explainers.
    """
    def __init__(self, model, features, target_column=None):
        self.model = model
        self.features = features
        self.target_column = target_column
    
    def predict(self, X):
        """Standard prediction interface"""
        # Ensure X contains only required features, in the right order
        if isinstance(X, pd.DataFrame):
            # Filter to only include necessary features
            required_features = [f for f in self.features if f != self.target_column]
            available_features = [f for f in required_features if f in X.columns]
            
            if len(available_features) != len(required_features):
                missing = set(required_features) - set(available_features)
                print(f"Warning: Missing features for prediction: {missing}")
            
            X_filtered = X[available_features]
            return self.model.predict(X_filtered)
        else:
            # Handle numpy arrays or other formats
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """Probabilistic prediction interface if model supports it"""
        if not hasattr(self.model, 'predict_proba'):
            # Fall back to regular prediction if probabilities not available
            preds = self.predict(X)
            # Convert to one-hot encoding for binary classification
            if len(np.unique(preds)) <= 2:
                return np.column_stack((1-preds, preds))
            else:
                # For multiclass, return dummy probabilities
                classes = len(np.unique(preds))
                probs = np.zeros((len(preds), classes))
                for i, p in enumerate(preds):
                    probs[i, int(p)] = 1.0
                return probs
            
        # Filter features as in predict method
        if isinstance(X, pd.DataFrame):
            required_features = [f for f in self.features if f != self.target_column]
            X_filtered = X[[f for f in required_features if f in X.columns]]
            return self.model.predict_proba(X_filtered)
        else:
            return self.model.predict_proba(X)
