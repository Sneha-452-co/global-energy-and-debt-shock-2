import pickle
import pandas as pd
import numpy as np
import streamlit as st
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Energy Predictor", layout="centered")

st.title("🌍 Global Energy & Debt Shock")
st.markdown("Predict energy/emission values using ML model")

# -------------------------------
# Paths (FIXED → relative paths)
# -------------------------------
SCALER_PATH = "notebook/scaler.pkl"
MODEL_PATH = "notebook/model.pkl"
ENCODER_PATH = "notebook/encoder.pkl"

# -------------------------------
# Load Models (cached)
# -------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ File not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

scaler = load_model('/workspaces/global-energy-and-debt-shock-2/notebook/scaler.pkl')
model = load_model('/workspaces/global-energy-and-debt-shock-2/notebook/model.pkl')
encoder = load_model('/workspaces/global-energy-and-debt-shock-2/notebook/encoder.pkl')

# -------------------------------
# Prediction Function
# -------------------------------
def predict_energy(country, year, oil, gas, coal, renew):

    if scaler is None or model is None or encoder is None:
        return None

    try:
        # Encode country
        country_enc = encoder.transform([country])[0]
    except Exception:
        st.error("❌ Country not present in training data")
        return None

    # Feature order MUST match training
    data = pd.DataFrame({
        "Country": [country_enc],
        "Year": [year],
        "oilcons_ej": [oil],
        "gascons_ej": [gas],
        "coalcons_ej": [coal],
        "renewables_ej": [renew]
    })

    try:
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
        return prediction
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None


# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📊 Input Parameters")

country = st.sidebar.selectbox(
    "Select Country",
    ["India", "United States", "China", "Japan", "Germany"]
)

year = st.sidebar.slider("Year", 1965, 2023, 2020)

oil = st.sidebar.number_input("Oil Consumption (EJ)", min_value=0.0, value=10.0)
gas = st.sidebar.number_input("Gas Consumption (EJ)", min_value=0.0, value=5.0)
coal = st.sidebar.number_input("Coal Consumption (EJ)", min_value=0.0, value=8.0)
renew = st.sidebar.number_input("Renewables (EJ)", min_value=0.0, value=3.0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🚀 Predict"):

    result = predict_energy(country, year, oil, gas, coal, renew)

    if result is not None:
        st.success(f"📈 Predicted Value: {result:.2f}")

        # Safe progress bar
        progress_val = min(max(result / 10000, 0), 1)
        st.progress(progress_val)

        st.info("Higher values indicate greater energy/emission output")

    else:
        st.error("❌ Prediction failed. Check model files.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("AI/ML Project: Global Energy & Debt Shock Analysis")