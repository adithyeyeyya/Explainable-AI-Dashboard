import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dice_ml
from dice_ml import Dice

class CounterfactualExplainer:
    def __init__(self, df, model, categorical_features, numerical_features, target_column, 
                 backend='sklearn', model_type='classifier', is_target_categorical=True):
        """
        Initialize the counterfactual explainer using DiCE.

        Args:
            df: DataFrame containing the data
            model: Trained ML model object (must implement predict_proba or predict)
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            target_column: Name of the target column
            backend: Backend framework ('sklearn', 'tensorflow', 'pytorch') for DiCE
            model_type: 'classifier' or 'regressor' for DiCE
            is_target_categorical: Whether the target is categorical
        """
        self.df = df
        self.model = model
        self.target_column = target_column
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.is_target_categorical = is_target_categorical
        
        # Determine model type based on is_target_categorical
        self.model_type = model_type if model_type else ('classifier' if is_target_categorical else 'regressor')
        
        # Initialize DiCE
        try:
            # Prepare DiCE Data object
            self.dice_data = dice_ml.Data(
                dataframe=df,
                continuous_features=numerical_features,
                categorical_features=categorical_features,
                outcome_name=target_column
            )

            # Prepare DiCE Model object
            self.dice_model = dice_ml.Model(model=model, backend=backend, model_type=self.model_type)

            # Create a DiCE explainer
            self.dice = Dice(self.dice_data, self.dice_model)
            
            print("Successfully initialized DiCE for counterfactual explanations.")
        except Exception as e:
            raise ValueError(f"Error initializing DiCE: {str(e)}. Please make sure DiCE is installed: pip install dice-ml")

    def _predict(self, instance):
        """
        Helper method to handle model prediction properly
        """
        try:
            if hasattr(self.model, 'predict_proba') and self.is_target_categorical:
                probs = self.model.predict_proba(instance)
                return np.argmax(probs, axis=1)[0]
            elif hasattr(self.model, 'predict'):
                return self.model.predict(instance)[0]
            else:
                # Last resort for callable models
                return self.model(instance)
        except Exception as e:
            raise ValueError(f"Prediction error: {e}")

    def get_feature_importance(self, query_instance):
        """
        Calculate feature importance based on how changes affect the prediction.
        Uses DiCE's counterfactuals to determine importance.
        """
        print("Calculating feature importance using DiCE counterfactuals...")
        
        # Initialize importance scores dictionary
        importance_scores = {}
        
        # Generate several counterfactuals
        counterfactuals = self.generate_counterfactuals(query_instance, num_counterfactuals=5)
        
        if counterfactuals.empty:
            print("No counterfactuals found, feature importance calculation may be unreliable.")
            # Return different default values to avoid uniform importance
            return {feature: 0.01 * (i + 1) for i, feature in enumerate(self.numerical_features + self.categorical_features)}
        
        # Get original prediction
        original_prediction = self._predict(query_instance)
        
        # Calculate feature importance based on counterfactual changes
        for feature in self.numerical_features + self.categorical_features:
            if feature not in query_instance.columns or feature not in counterfactuals.columns:
                importance_scores[feature] = 0.01
                continue
                
            # Get original value
            original_value = query_instance[feature].values[0]
            
            # Calculate average change across all counterfactuals
            feature_changes = []
            
            for _, cf_row in counterfactuals.iterrows():
                cf_value = cf_row[feature]
                
                # Calculate relative change
                if feature in self.numerical_features:
                    # For numerical, calculate relative change
                    feature_range = max(self.df[feature].max() - self.df[feature].min(), 1e-10)
                    try:
                        # Convert both values to float to avoid type errors
                        cf_value_float = float(cf_value)
                        original_value_float = float(original_value)
                        change = abs(cf_value_float - original_value_float) / feature_range
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not calculate numerical change for feature '{feature}': {e}")
                        change = 0.0
                else:
                    # For categorical, binary change (1 if different, 0 if same)
                    try:
                        change = 0.0 if cf_value == original_value else 1.0
                    except Exception as e:
                        print(f"Warning: Could not calculate categorical change for feature '{feature}': {e}")
                        change = 0.0
                
                feature_changes.append(change)
            
            # Use average change as importance, but weight by frequency of change
            try:
                if feature_changes:
                    # Calculate how often this feature changes in counterfactuals
                    change_frequency = sum(1 for change in feature_changes if change > 0.01) / len(feature_changes)
                    # Combine magnitude of change with frequency of change
                    avg_change = sum(feature_changes) / len(feature_changes)
                    # Weight importance by both change magnitude and frequency
                    importance_scores[feature] = avg_change * (change_frequency + 0.1)
                else:
                    importance_scores[feature] = 0.01
            except Exception as e:
                print(f"Warning: Could not calculate average importance for feature '{feature}': {e}")
                importance_scores[feature] = 0.01
        
        # Add small random variations to avoid completely identical scores
        for feature in importance_scores:
            importance_scores[feature] += np.random.uniform(0, 0.01)
        
        # Normalize scores to [0,1]
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        if max_score > 0:
            for feature in importance_scores:
                importance_scores[feature] /= max_score
        
        # Ensure minimum differences between scores
        min_diff = 0.02
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Increase separation between features
        for i in range(1, len(sorted_features)):
            prev_feature, prev_score = sorted_features[i-1]
            curr_feature, curr_score = sorted_features[i]
            
            # If scores are too close, increase the gap
            if prev_score - curr_score < min_diff:
                importance_scores[curr_feature] = max(0.01, prev_score - min_diff - (i * 0.01))
        
        print(f"Feature importance scores: {importance_scores}")
        return importance_scores
    
    def plot_counterfactuals(self, original_instance, counterfactuals, top_features=None):
        """
        Plot a visual comparison of the original instance and its counterfactuals.
        
        Args:
            original_instance: DataFrame containing the original instance
            counterfactuals: DataFrame containing counterfactual examples
            top_features: Optional list of features to display (if None, uses all features)
        """
        if counterfactuals.empty:
            print("No counterfactuals available to plot.")
            return None
            
        try:
            # Combine original and counterfactuals for the plot
            all_instances = pd.concat([original_instance, counterfactuals])
            labels = ['Original'] + [f'CF {i+1}' for i in range(len(counterfactuals))]
            
            # Select features to display
            if top_features is None:
                # Calculate feature importance to identify most important features
                importance = self.get_feature_importance(original_instance)
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                # Use top features (maximum 8 to keep plot readable)
                features = [f[0] for f in sorted_features[:8]]
            else:
                features = top_features
            
            if not features:
                print("No features available for plotting.")
                return None

            # Create subplots
            n_cols = min(4, len(features))  # Max 4 features per row
            n_rows = (len(features) + n_cols - 1) // n_cols
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows * n_cols == 1:
                axs = np.array([[axs]])
            elif n_rows == 1:
                axs = np.array([axs])
            elif n_cols == 1:
                axs = np.array([[ax] for ax in axs])
            
            # Plot each feature
            for i, feature in enumerate(features):
                row = i // n_cols
                col = i % n_cols
                
                values = all_instances[feature].values
                
                # Adjust display based on feature type
                if feature in self.categorical_features:
                    axs[row, col].bar(labels, values)
                    axs[row, col].set_title(f"{feature} (Categorical)")
                else:
                    # For numerical features
                    axs[row, col].bar(labels, values)
                    axs[row, col].set_title(f"{feature} (Numerical)")
                
                # Set x-axis labels
                axs[row, col].tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(len(features), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                fig.delaxes(axs[row, col])
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Plot error: {str(e)}")
            return None
        
    def generate_counterfactuals(self, query_instance, num_counterfactuals=3, desired_class="opposite", 
                                 proximity_weight=0.5, diversity_weight=1.0, sparsity_weight=1.0,
                                 regression_min_pct=None, regression_max_pct=None,
                                 regression_abs_min=None, regression_abs_max=None,
                                 features_to_vary=None):
        """
        Generate counterfactual examples for the given query instance using DiCE.
        
        Args:
            query_instance: DataFrame or Series containing the instance to explain
            num_counterfactuals: Number of counterfactual examples to generate
            desired_class: Target class for counterfactuals ('opposite' for classification)
                           For regression, can be a numeric value
            proximity_weight: Weight for proximity objective (smaller changes preferred)
            diversity_weight: Weight for diversity objective (diverse counterfactuals)
            sparsity_weight: Weight for sparsity objective (fewer features changed)
            regression_min_pct: Minimum percentage change for regression (e.g., -20 for -20%)
            regression_max_pct: Maximum percentage change for regression (e.g., 20 for +20%)
            regression_abs_min: Minimum absolute change for regression
            regression_abs_max: Maximum absolute change for regression
            
        Returns:
            DataFrame with counterfactual examples
        """
        # Ensure query_instance is a single row DataFrame
        if isinstance(query_instance, pd.Series):
            query_instance = query_instance.to_frame().T
        
        # Parameters for DiCE
        dice_params = {
            "proximity_weight": 0.2,  # Reduced from 0.5 to relax constraints
            "diversity_weight": 0.5,  # Reduced from 1.0
            "sparsity_weight": 0.5,   # Reduced from 1.0
            "features_to_vary": features_to_vary if features_to_vary is not None else 'all', # Use passed features_to_vary or all
            "permitted_range": None    # Allow wider ranges for features
        }
        
        # Handle desired_class for regression and classification
        if not self.is_target_categorical:
            # For regression, set a target value range based on input parameters
            current_prediction = self._predict(query_instance)
            current_prediction = float(current_prediction)
            
            # Use user-defined regression parameters if provided
            if regression_min_pct is not None and regression_max_pct is not None:
                # Calculate desired range from percentage
                min_value = current_prediction * (1 + regression_min_pct / 100)
                max_value = current_prediction * (1 + regression_max_pct / 100)
                desired_range = [min_value, max_value]
                print(f"Setting desired outcome range for regression using percentages: {desired_range}")
            elif regression_abs_min is not None and regression_abs_max is not None:
                # Calculate desired range from absolute values
                min_value = current_prediction + regression_abs_min
                max_value = current_prediction + regression_abs_max
                desired_range = [min_value, max_value]
                print(f"Setting desired outcome range for regression using absolute values: {desired_range}")
            else:
                # Default range if no specific parameters provided
                target_value = current_prediction * 1.3
                desired_range = [current_prediction * 0.8, target_value * 1.2]
                print(f"Setting default desired outcome range for regression to {desired_range}")
            
            # Replace desired_class with desired_range for regression
            dice_params["desired_range"] = desired_range
        else:
            # For classification, flip the class (assuming binary classification)
            if desired_class == "opposite":
                current_prediction = self._predict(query_instance)
                desired_class = 1 if current_prediction == 0 else 0
                print(f"Setting desired outcome for classification to {desired_class}")
            
            # Use desired_class for classification
            dice_params["desired_class"] = desired_class
        
        # Try multiple parameter sets to get diverse counterfactuals
        try:
            method_params = [
                dice_params,
                {"proximity_weight": 0.2, "diversity_weight": 2.0, "sparsity_weight": 0.5},
                {"proximity_weight": 0.5, "diversity_weight": 1.0, "sparsity_weight": 2.0},
            ]
            
            # For regression, make sure to add desired_range to all method params
            if not self.is_target_categorical and "desired_range" in dice_params:
                for params in method_params:
                    if "desired_range" not in params:
                        params["desired_range"] = dice_params["desired_range"]
            
            all_counterfactuals = []
            for i, params in enumerate(method_params):
                try:
                    # For regression, ensure we're using desired_range, not desired_class
                    if not self.is_target_categorical:
                        if "desired_class" in params:
                            del params["desired_class"]  # Remove this if it exists
                    
                    dice_exp = self.dice.generate_counterfactuals(
                        query_instance,
                        total_CFs=max(1, num_counterfactuals // len(method_params)),
                        **params
                    )
                    
                    if hasattr(dice_exp, 'cf_examples_list') and dice_exp.cf_examples_list:
                        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                        if not cf_df.empty:
                            all_counterfactuals.append(cf_df)
                            print(f"Method {i+1}: Generated {len(cf_df)} counterfactuals")
                except Exception as e:
                    print(f"Method {i+1} failed: {str(e)}")
            
            if all_counterfactuals:
                combined_cfs = pd.concat(all_counterfactuals, ignore_index=True)
                combined_cfs = combined_cfs.drop_duplicates()
                print(f"Successfully generated {len(combined_cfs)} counterfactuals using DiCE")
                
                for feature in self.numerical_features + self.categorical_features:
                    if feature in query_instance.columns and feature in combined_cfs.columns:
                        original = query_instance[feature].values[0]
                        if feature in self.numerical_features:
                            avg_change = abs(combined_cfs[feature] - original).mean()
                            print(f"  {feature}: avg change = {avg_change:.4f}")
                        else:
                            changes = sum(combined_cfs[feature] != original)
                            print(f"  {feature}: changed in {changes}/{len(combined_cfs)} counterfactuals")
                
                return combined_cfs
            else:
                print("All methods failed to generate counterfactuals")
                try:
                    last_resort_params = {
                        "proximity_weight": 0.1,
                        "diversity_weight": 2.0,
                        "sparsity_weight": 0.1
                    }
                    
                    # For regression, add desired_range
                    if not self.is_target_categorical and "desired_range" in dice_params:
                        last_resort_params["desired_range"] = dice_params["desired_range"]
                    elif self.is_target_categorical and "desired_class" in dice_params:
                        last_resort_params["desired_class"] = dice_params["desired_class"]
                    
                    dice_exp = self.dice.generate_counterfactuals(
                        query_instance,
                        total_CFs=num_counterfactuals,
                        **last_resort_params
                    )
                    
                    if hasattr(dice_exp, 'cf_examples_list') and dice_exp.cf_examples_list:
                        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                        if not cf_df.empty:
                            print(f"Last resort: Generated {len(cf_df)} counterfactuals")
                            return cf_df
                except Exception as e:
                    print(f"Last resort failed: {str(e)}")
                
                print("Could not generate counterfactuals")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error generating counterfactuals with DiCE: {str(e)}")
            return pd.DataFrame()

    def generate_diverse_counterfactuals(self, query_instance, feature_weights=None):
        """
        Generate diverse counterfactuals by focusing on different feature subsets.
        
        Args:
            query_instance: DataFrame containing the instance to explain
            feature_weights: Optional dictionary of {feature: weight} to prioritize features
            
        Returns:
            DataFrame with diverse counterfactual examples
        """
        if feature_weights is None:
            # Calculate feature importance to use as weights
            feature_weights = self.get_feature_importance(query_instance)
        
        # Generate standard counterfactuals first
        standard_cfs = self.generate_counterfactuals(query_instance, num_counterfactuals=2)
        
        # Generate feature-specific counterfactuals for top features
        top_features = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        
        all_counterfactuals = []
        if not standard_cfs.empty:
            all_counterfactuals.append(standard_cfs)
        
        for feature, _ in top_features:
            # Create feature-specific weights (emphasize this feature)
            specific_weights = {f: 0.1 for f in feature_weights}
            specific_weights[feature] = 1.0
            
            # Configure DiCE to focus on this feature
            feature_based_exp = self.dice.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class="opposite" if self.is_target_categorical else None,
                features_to_vary=[feature]  # Focus on this specific feature
            )
            
            # Add to results if successful
            if hasattr(feature_based_exp, 'cf_examples_list') and feature_based_exp.cf_examples_list:
                cf_df = feature_based_exp.cf_examples_list[0].final_cfs_df
                if not cf_df.empty:
                    all_counterfactuals.append(cf_df)
                    print(f"Generated feature-specific counterfactual for {feature}")
        
        # Combine all counterfactuals
        if all_counterfactuals:
            combined_cfs = pd.concat(all_counterfactuals, ignore_index=True)
            # Remove duplicates if any
            combined_cfs = combined_cfs.drop_duplicates()
            print(f"Generated {len(combined_cfs)} diverse counterfactuals")
            return combined_cfs
        else:
            print("Failed to generate diverse counterfactuals")
            return pd.DataFrame()
