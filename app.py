import streamlit as st
from model_utils import predict, FEATURE_NAMES

st.set_page_config(page_title="🧠 Parkinson’s Voice Test", layout="centered")

st.title("🧪 Parkinson’s Voice Detection App")
st.markdown("Enter voice feature values manually to get a UPDRS severity score and Parkinson’s risk assessment.")

# Input form
st.header("🎛️ Voice Feature Inputs")
features = []

for name in FEATURE_NAMES:
    val = st.number_input(f"{name}", value=0.0, step=0.001, format="%.5f")
    features.append(val)

# Run prediction
if st.button("🔍 Run Prediction"):
    try:
        updrs_score, severity_label = predict(features)

        st.success(f"🎯 **Predicted UPDRS Score:** {updrs_score:.2f}")
        st.info(f"🚨 **Predicted Severity:** {severity_label}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
