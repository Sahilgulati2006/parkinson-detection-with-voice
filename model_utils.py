import joblib
import numpy as np
import pandas as pd

# Load models and scaler
reg_model = joblib.load("updrs_regression_model.pkl")
clf_model = joblib.load("parkinsons_classifier_model.pkl")
scaler = joblib.load("voice_scaler.pkl")

# The 16 voice features used during training (order matters!)
FEATURE_NAMES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "PPE"
]

def predict(features_list):
    """
    Takes a list of 16 voice features in the correct order.
    Returns: (predicted_UPDRS_score, severity_class_label)
    """
    # Convert to DataFrame with correct column names
    input_df = pd.DataFrame([features_list], columns=FEATURE_NAMES)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict with both models
    updrs_score = reg_model.predict(input_scaled)[0]
    class_output = clf_model.predict(input_scaled)[0]
    severity_label = "High Risk of Parkinson’s" if class_output == 1 else "Low Risk (Likely Healthy)"

    return updrs_score, severity_label
