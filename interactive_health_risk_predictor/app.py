import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Locate base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load scaler
try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error(f"❌ Scaler not found at: {SCALER_PATH}")
    st.stop()

# Load model
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
except FileNotFoundError:
    st.error(f"❌ Model not found at: {MODEL_PATH}")
    st.stop()

pipeline = model_data.get("pipeline", None)
feature_names = model_data.get("feature_names", [])

if pipeline is None:
    st.error("❌ Model pipeline not found.")
    st.stop()

# --------------------------------------------------
# Streamlit Config
st.set_page_config(page_title="Health Risk Predictor", page_icon="🩺", layout="centered")

# Title with emoji
st.title("🩺 Interactive Health Risk Predictor")
st.markdown("### Predict your diabetes risk instantly!")
st.markdown("This tool uses **Machine Learning** to analyze your health metrics and predict whether you are at **Low Risk** or **High Risk** of diabetes.")

# Divider
st.divider()

# Side Panel Info
with st.sidebar:
    st.header("📊 About the Model")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Features Used:**")
    for f in feature_names:
        st.write(f"• {f}")
    st.info("💡 Tip: Enter realistic values for accurate predictions.")

# Input Form
st.subheader("Enter Your Health Data")
user_input = []
with st.form("health_form"):
    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        val = cols[i % 2].number_input(f"{feature}", value=0.0)
        user_input.append(val)
    submitted = st.form_submit_button("🔍 Predict Risk")

# Prediction
if submitted:
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = pipeline.predict(input_scaled)[0]
    proba = pipeline.predict_proba(input_scaled)[0][prediction]

    # Animated progress bar
    st.write("### Analyzing your data...")
    progress = st.progress(0)
    for i in range(1, 101):
        progress.progress(i)

    # Show result
    if prediction == 1:
        st.error(f"🚨 **High Risk Detected** ({proba*100:.1f}% probability)")
    else:
        st.success(f"✅ **Low Risk** ({proba*100:.1f}% probability)")

    # Extra insights
    st.subheader("📈 Risk Probability Breakdown")
    risk_df = pd.DataFrame({
        "Risk Type": ["Low Risk", "High Risk"],
        "Probability": pipeline.predict_proba(input_scaled)[0]
    })
    st.bar_chart(risk_df.set_index("Risk Type"))

# Footer
st.divider()
st.caption("Built using Streamlit & Scikit-learn")