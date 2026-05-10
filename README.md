# Explainable-AI-Dashboard

An interactive Explainable AI (XAI) dashboard that enables users to upload machine learning code and datasets to generate model insights, interpretability visualizations, and counterfactual explanations.

Features
Upload custom ML models and datasets
Generate explainability insights automatically
Counterfactual explanations using DiCE
Feature importance visualization
Interactive dashboard interface
Supports multiple machine learning workflows
Technologies Used
Python
Streamlit / Flask / FastAPI (whichever you used)
Scikit-learn
DiCE (Diverse Counterfactual Explanations)
Pandas
NumPy
Matplotlib / Plotly
How It Works
Upload machine learning model/code
Upload dataset
Dashboard analyzes model behavior
Generates:
Feature importance
Prediction explanations
Counterfactual examples
Model interpretability insights
Counterfactual Explanations with DiCE

The system uses DiCE to generate diverse counterfactual explanations, helping users understand:

What minimal changes alter predictions
Model decision boundaries
Feature sensitivity

Example:

“If income increased by $5,000 and loan amount decreased by $2,000, the prediction changes from rejected to approved.”
