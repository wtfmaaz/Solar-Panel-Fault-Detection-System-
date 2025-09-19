import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import time
import os
import joblib

# --- Page config ---
st.set_page_config(page_title="☀️ Solar Panel Fault Detection", layout="wide")

# --- Models ---
MODEL_DIR = "models"
TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_rf.pkl")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_thermal.h5")

# Load models if available
@st.cache_resource
def load_tabular_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_cnn_model(path):
    from tensorflow.keras.models import load_model
    if os.path.exists(path):
        return load_model(path)
    return None

tabular_model = load_tabular_model(TABULAR_MODEL_PATH)
cnn_model = load_cnn_model(CNN_MODEL_PATH)

# --- Sidebar Configuration ---
st.sidebar.header("Settings / Config")
use_backend = st.sidebar.checkbox("Use FastAPI backend?", value=False)
backend_url = st.sidebar.text_input("Backend /infer endpoint (include /infer)", value="")

st.sidebar.markdown("**Optional:** Download models if missing")

# --- Main Layout ---
st.title("☀️ Solar Panel Fault Detection & Monitoring")
tabs = st.tabs(["Manual / CSV Input", "Thermal Image Prediction", "Live Simulation", "History & Logs"])

# --- Tab 1: Manual / CSV Input ---
with tabs[0]:
    st.subheader("Sensor Input")
    mode = st.radio("Input mode:", ["Manual reading", "CSV upload"], index=0)

    if mode == "Manual reading":
        irradiance = st.number_input("Irradiance (W/m²)", value=800.0)
        ambient_temp = st.number_input("Ambient temperature (°C)", value=30.0)
        panel_temp = st.number_input("Panel temperature (°C)", value=35.0)
        voltage = st.number_input("Voltage (V)", value=18.5)
        current = st.number_input("Current (A)", value=0.85)

        if st.button("Predict Fault"):
            row = {
                "irradiance_wpm2": irradiance,
                "ambient_temp_c": ambient_temp,
                "panel_temp_c": panel_temp,
                "voltage_v": voltage,
                "current_a": current,
            }
            # Derived features
            row["power_w"] = row["voltage_v"] * row["current_a"]
            row["efficiency"] = row["power_w"]/(row["irradiance_wpm2"]+1e-6)
            row["temp_delta"] = row["panel_temp_c"] - row["ambient_temp_c"]
            row["residual_power"] = 0.0

            # Backend call
            if use_backend and backend_url:
                try:
                    resp = requests.post(backend_url, json=row, timeout=10, verify=False)
                    st.json(resp.json())
                except Exception as e:
                    st.error(f"Backend request failed: {e}")
            else:
                if tabular_model:
                    feat_list = ['irradiance_wpm2','panel_temp_c','voltage_v','current_a','power_w','efficiency','temp_delta','residual_power']
                    x = np.array([[row[f] for f in feat_list]])
                    pred = tabular_model.predict(x)[0]
                    prob = tabular_model.predict_proba(x).max() if hasattr(tabular_model, "predict_proba") else None
                    st.success(f"Predicted Fault: {pred} ({prob:.2f} confidence)" if prob else f"Predicted Fault: {pred}")
                else:
                    st.warning("Tabular model not found. Upload models in `models/` folder.")

    else:
        up = st.file_uploader("Upload CSV with columns (timestamp, irradiance_wpm2, ambient_temp_c, panel_temp_c, voltage_v, current_a, label(optional))", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.dataframe(df.head())
            if st.button("Batch Predict"):
                df['power_w'] = df['voltage_v'] * df['current_a']
                df['efficiency'] = df['power_w']/(df['irradiance_wpm2']+1e-6)
                df['temp_delta'] = df['panel_temp_c'] - df['ambient_temp_c']
                df['residual_power'] = df['power_w'] - df['power_w'].rolling(30, min_periods=1).mean()

                feat_list = ['irradiance_wpm2','panel_temp_c','voltage_v','current_a','power_w','efficiency','temp_delta','residual_power']
                if tabular_model:
                    df['pred'] = tabular_model.predict(df[feat_list])
                    st.dataframe(df.head(20))
                    csv = df.to_csv(index=False).encode()
                    st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv")
                else:
                    st.error("Tabular model not found.")

# --- Tab 2: Thermal Image ---
with tabs[1]:
    st.subheader("Thermal / IR Image Prediction")
    img_up = st.file_uploader("Upload thermal image (PNG/JPG)", type=["png","jpg","jpeg"])
    if img_up is not None:
        image = Image.open(img_up).convert("L").resize((64,64))
        st.image(image, caption="Uploaded image (grayscale)")

        arr = np.array(image)[...,np.newaxis]/255.0
        if cnn_model is not None and st.button("Predict Image Fault"):
            p = cnn_model.predict(np.expand_dims(arr, 0)).argmax(axis=1)[0]
            st.success(f"Predicted Image Fault Index: {p} — map to labels used in training")
        elif cnn_model is None:
            st.warning("CNN model not loaded.")

# --- Tab 3: Live Simulation / Demo ---
with tabs[2]:
    st.subheader("Live Fault Simulation")
    simulate = st.button("Start Live Simulation")
    placeholder = st.empty()
    if simulate:
        for i in range(10):  # simulate 10 readings
            irradiance = 800 + np.random.randint(-50,50)
            panel_temp = 35 + np.random.rand()*5
            ambient_temp = 30 + np.random.rand()*5
            voltage = 18.5 + np.random.rand()*0.5
            current = 0.85 + np.random.rand()*0.1
            power = voltage * current
            efficiency = power/(irradiance+1e-6)
            temp_delta = panel_temp - ambient_temp
            pred = "Normal"
            if i%5==0: pred="Shading"
            placeholder.metric(label="Fault Prediction", value=pred)
            time.sleep(1)

# --- Tab 4: History / Logs ---
with tabs[3]:
    st.subheader("Inference Log / History")
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Add Test Record"):
        sample = {
            "irradiance_wpm2": 800,
            "panel_temp_c": 35,
            "voltage_v": 18.5,
            "current_a": 0.85
        }
        sample["power_w"] = sample["voltage_v"] * sample["current_a"]
        sample["efficiency"] = sample["power_w"] / (sample["irradiance_wpm2"]+1e-6)
        sample["temp_delta"] = sample["panel_temp_c"] - 30
        sample["residual_power"] = 0.0
        st.session_state.history.append(sample)

    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history).tail(20))
    else:
        st.info("No inferences yet. Use inputs above or add test records.")










































