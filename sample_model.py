from sklearn.ensemble import RandomForestClassifier

def create_model(X_train, y_train):
    """
    Create and train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        A trained sklearn model
    """
    # Create a simple random forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model