# app.py
# -*- coding: utf-8 -*-

import os
import time
import requests
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ML / DL imports
import joblib
from tensorflow.keras.models import load_model

# Config
MODEL_DIR = "models"
TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_rf.pkl")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_thermal.h5")
CLASSES = ["Normal", "Soiling", "Shading", "Hotspot"]

# Ensure models exist
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load tabular model
@st.cache_resource
def load_tabular_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load tabular model: {e}")
        return None

# --- Load CNN model
@st.cache_resource
def load_cnn_model(path):
    try:
        return load_model(path)
    except Exception as e:
        st.info("CNN model not loaded (optional).")
        return None

tabular_model = load_tabular_model(TABULAR_MODEL_PATH) if os.path.exists(TABULAR_MODEL_PATH) else None
cnn_model = load_cnn_model(CNN_MODEL_PATH) if os.path.exists(CNN_MODEL_PATH) else None

# --- Streamlit UI ---
st.set_page_config(page_title="Solar Panel Fault Monitor", layout="wide")
st.title("☀️ Solar Panel Fault Detection & Monitoring")

# Layout: left inputs, right outputs
left, right = st.columns([1,1])

with left:
    st.subheader("Sensor Input / CSV Upload")
    mode = st.radio("Input mode:", ["Manual reading", "CSV upload"], index=0)

    if mode == "Manual reading":
        irradiance = st.number_input("Irradiance (W/m²)", value=800.0)
        ambient_temp = st.number_input("Ambient temperature (°C)", value=30.0)
        panel_temp = st.number_input("Panel temperature (°C)", value=35.0)
        voltage = st.number_input("Voltage (V)", value=18.5)
        current = st.number_input("Current (A)", value=0.85)

        if st.button("Predict (Local)"):
            row = {
                "irradiance_wpm2": irradiance,
                "ambient_temp_c": ambient_temp,
                "panel_temp_c": panel_temp,
                "voltage_v": voltage,
                "current_a": current
            }
            # Derived features
            row["power_w"] = row["voltage_v"] * row["current_a"]
            row["efficiency"] = row["power_w"] / (row["irradiance_wpm2"] + 1e-6)
            row["temp_delta"] = row["panel_temp_c"] - row["ambient_temp_c"]
            row["residual_power"] = 0.0  # single reading

            if tabular_model:
                features = ["irradiance_wpm2","panel_temp_c","voltage_v","current_a","power_w",
                            "efficiency","temp_delta","residual_power"]
                x = np.array([[row[f] for f in features]])
                pred = tabular_model.predict(x)[0]
                prob = tabular_model.predict_proba(x).max() if hasattr(tabular_model, "predict_proba") else None
                st.success(f"Predicted: {pred} ({prob:.2f} confidence)" if prob else f"Predicted: {pred}")
            else:
                st.error("Tabular model not found. Please upload or download.")

    else:
        up = st.file_uploader("Upload CSV with required columns", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            st.dataframe(df.head())
            if st.button("Batch Predict"):
                # Compute derived features
                df["power_w"] = df["voltage_v"] * df["current_a"]
                df["efficiency"] = df["power_w"] / (df["irradiance_wpm2"] + 1e-6)
                df["temp_delta"] = df["panel_temp_c"] - df["ambient_temp_c"]
                df["rolling_power_30s"] = df["power_w"].rolling(window=30, min_periods=1).mean()
                df["residual_power"] = df["power_w"] - df["rolling_power_30s"]

                features = ["irradiance_wpm2","panel_temp_c","voltage_v","current_a","power_w",
                            "efficiency","temp_delta","residual_power"]
                if tabular_model:
                    df["predicted_label"] = tabular_model.predict(df[features])
                    st.dataframe(df.head(50))
                else:
                    st.error("Tabular model not loaded.")

    st.markdown("---")
    st.subheader("Thermal / IR Image Prediction (Optional)")
    img_up = st.file_uploader("Upload thermal image", type=["png","jpg","jpeg"])
    if img_up is not None:
        image = Image.open(img_up).convert("L").resize((64,64))
        st.image(image, caption="Uploaded Image (grayscale)")
        arr = np.array(image)[...,np.newaxis] / 255.0
        if cnn_model is not None and st.button("Predict Image"):
            idx = cnn_model.predict(np.expand_dims(arr,0)).argmax(axis=1)[0]
            st.success(f"Predicted Image Class: {CLASSES[idx]}")

with right:
    st.subheader("Inference Log / History")
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Add Sample Record"):
        sample = {
            "irradiance_wpm2": 800,
            "panel_temp_c": 35,
            "voltage_v": 18.5,
            "current_a": 0.85
        }
        sample["power_w"] = sample["voltage_v"] * sample["current_a"]
        sample["efficiency"] = sample["power_w"] / (sample["irradiance_wpm2"] + 1e-6)
        sample["temp_delta"] = sample["panel_temp_c"] - 30
        sample["residual_power"] = 0.0

        if tabular_model:
            features = ["irradiance_wpm2","panel_temp_c","voltage_v","current_a","power_w",
                        "efficiency","temp_delta","residual_power"]
            x = np.array([[sample[f] for f in features]])
            pred = tabular_model.predict(x)[0]
            prob = tabular_model.predict_proba(x).max() if hasattr(tabular_model,"predict_proba") else None
            st.session_state.history.append({"ts": time.time(), "pred": pred, "conf": float(prob) if prob else None})
            st.success(f"{pred} ({prob:.2f} confidence)" if prob else f"{pred}")
        else:
            st.info("Tabular model not loaded.")

    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history).tail(20))
    else:
        st.info("No inferences yet. Use inputs on left and click Predict or Add Sample Record.")

st.markdown("---")
st.info("Tip: For device integration, deploy FastAPI as backend. Streamlit is for monitoring & UI.")











































