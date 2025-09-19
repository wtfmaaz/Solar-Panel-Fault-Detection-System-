# app.py - Streamlit Frontend for Solar Fault Detection

import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import requests
from PIL import Image
import time

st.set_page_config(page_title="Solar Fault Monitor", layout="wide")

# ----------------- Model Paths -----------------
MODEL_DIR = "models"
TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_rf.pkl")
CNN_MODEL_PATH     = os.path.join(MODEL_DIR, "cnn_thermal.h5")

# ----------------- Load Models -----------------
@st.cache_resource
def load_tabular_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Tabular model not found: {e}")
        return None

@st.cache_resource
def load_cnn_model(path):
    try:
        from tensorflow.keras.models import load_model
        return load_model(path)
    except Exception as e:
        st.info("CNN model not loaded (optional).")
        return None

tabular_model = load_tabular_model(TABULAR_MODEL_PATH) if os.path.exists(TABULAR_MODEL_PATH) else None
cnn_model = load_cnn_model(CNN_MODEL_PATH) if os.path.exists(CNN_MODEL_PATH) else None

# ----------------- Sidebar Config -----------------
st.sidebar.header("Configuration")
use_backend = st.sidebar.checkbox("Use remote FastAPI backend", value=False)
backend_url = st.sidebar.text_input(
    "Backend /infer endpoint (if using FastAPI)",
    value="",
    help="Example: https://<ngrok-id>.ngrok-free.app/infer"
)

# ----------------- Main UI -----------------
st.title("☀️ Solar Panel Fault Detection — Streamlit")

left, right = st.columns([1,1])

with left:
    st.subheader("Sensor Input")
    mode = st.radio("Input mode:", ["Manual Reading", "Upload CSV"], index=0)

    if mode == "Manual Reading":
        irr = st.number_input("Irradiance (W/m²)", value=800.0)
        amb = st.number_input("Ambient Temp (°C)", value=30.0)
        panel = st.number_input("Panel Temp (°C)", value=35.0)
        volt = st.number_input("Voltage (V)", value=18.5)
        curr = st.number_input("Current (A)", value=0.85)

        if st.button("Predict"):
            row = {
                "irradiance_wpm2": irr,
                "ambient_temp_c": amb,
                "panel_temp_c": panel,
                "voltage_v": volt,
                "current_a": curr,
                "power_w": volt * curr,
                "efficiency": (volt*curr)/(irr+1e-6),
                "temp_delta": panel - amb,
                "residual_power": 0.0
            }

            if use_backend and backend_url:
                try:
                    resp = requests.post(backend_url, json=row, timeout=10, verify=False)
                    st.json(resp.json())
                except Exception as e:
                    st.error(f"Backend request failed: {e}")
            else:
                if tabular_model:
                    feat_list = ['irradiance_wpm2','panel_temp_c','voltage_v',
                                 'current_a','power_w','efficiency','temp_delta','residual_power']
                    x = np.array([[row[f] for f in feat_list]])
                    pred = tabular_model.predict(x)[0]
                    prob = (tabular_model.predict_proba(x).max()
                            if hasattr(tabular_model,"predict_proba") else None)
                    st.success(f"Detected: {pred} ({prob:.2f} conf)" if prob else f"Detected: {pred}")
                else:
                    st.error("No local tabular model available.")

    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.dataframe(df.head())
            if st.button("Batch Predict"):
                df['power_w'] = df['voltage_v'] * df['current_a']
                df['efficiency'] = df['power_w']/(df['irradiance_wpm2']+1e-6)
                df['temp_delta'] = df['panel_temp_c'] - df['ambient_temp_c']
                df['rolling_power_30s'] = df['power_w'].rolling(window=30, min_periods=1).mean()
                df['residual_power'] = df['power_w'] - df['rolling_power_30s']
                feat_list = ['irradiance_wpm2','panel_temp_c','voltage_v','current_a',
                             'power_w','efficiency','temp_delta','residual_power']
                if tabular_model:
                    preds = tabular_model.predict(df[feat_list])
                    df['Prediction'] = preds
                    st.dataframe(df.head(30))
                else:
                    st.error("Model not found locally.")

    # CNN Image Inference
    st.subheader("Thermal Image Inference")
    img_up = st.file_uploader("Upload thermal/IR image", type=["png","jpg","jpeg"])
    if img_up is not None and cnn_model:
        image = Image.open(img_up).convert("L").resize((64,64))
        arr = np.array(image)[...,np.newaxis]/255.0
        pred_idx = cnn_model.predict(np.expand_dims(arr,0)).argmax(axis=1)[0]
        st.image(image, caption="Uploaded Image (Grayscale)")
        st.success(f"Predicted Image Class Index: {pred_idx}")

with right:
    st.subheader("Inference Log")
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.button("Add Sample Record"):
        sample = {"irradiance_wpm2": 800, "panel_temp_c": 35, "voltage_v": 18.5, "current_a": 0.85}
        sample["power_w"] = sample["voltage_v"]*sample["current_a"]
        sample["efficiency"] = sample["power_w"]/(sample["irradiance_wpm2"]+1e-6)
        sample["temp_delta"] = sample["panel_temp_c"] - 30
        sample["residual_power"] = 0.0
        if tabular_model:
            feat_list = ['irradiance_wpm2','panel_temp_c','voltage_v','current_a',
                         'power_w','efficiency','temp_delta','residual_power']
            x = np.array([[sample[f] for f in feat_list]])
            pred = tabular_model.predict(x)[0]
            conf = tabular_model.predict_proba(x).max() if hasattr(tabular_model,"predict_proba") else None
            st.session_state.history.append({"time": time.time(), "pred": pred, "conf": conf})
            st.success(f"{pred} (conf={conf:.2f})" if conf else f"{pred}")

    if st.session_state.history:
        st.write(pd.DataFrame(st.session_state.history).tail(20))
    else:
        st.info("No predictions yet.")











































