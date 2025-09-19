import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import joblib
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import requests
import os

# --- Page Setup ---
st.set_page_config(
    page_title="☀️ Solar Panel Fault Detection & Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("Settings / Config")
live_mode = st.sidebar.checkbox("Live Sensor Data (Simulation/Arduino)", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Fault Thresholds")
voltage_min = st.sidebar.number_input("Min Voltage (V)", 15.0)
current_min = st.sidebar.number_input("Min Current (A)", 0.5)

st.sidebar.markdown("---")
st.sidebar.header("Model / Backend Config")
use_backend = st.sidebar.checkbox("Use Remote FastAPI Backend", value=False)
backend_url = st.sidebar.text_input("Backend /infer endpoint", value="", help="Full URL including /infer")
raw_base = st.sidebar.text_input("Raw GitHub Base URL (optional)", help="Used to download models if not present")

# --- Models ---
MODEL_DIR = "models"
TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_rf.pkl")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_thermal.h5")

# Download models if missing
def download_if_missing(filename, raw_base):
    os.makedirs(MODEL_DIR, exist_ok=True)
    local = os.path.join(MODEL_DIR, filename)
    if os.path.exists(local): return local
    if not raw_base: return None
    url = raw_base.rstrip("/") + "/models/" + filename
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
        return local
    except Exception as e:
        st.warning(f"Could not download {filename}: {e}")
        return None

download_if_missing("tabular_rf.pkl", raw_base)
download_if_missing("cnn_thermal.h5", raw_base)

# Load tabular model
@st.cache_resource
def load_tabular_model(path):
    if os.path.exists(path):
        try: return joblib.load(path)
        except: st.error("Failed to load tabular model")
    return None

# Load CNN model
@st.cache_resource
def load_cnn_model(path):
    if os.path.exists(path):
        try:
            from tensorflow.keras.models import load_model
            return load_model(path)
        except: st.info("CNN model not loaded")
    return None

tabular_model = load_tabular_model(TABULAR_MODEL_PATH)
cnn_model = load_cnn_model(CNN_MODEL_PATH)

# --- Session State ---
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        "timestamp","irradiance","ambient_temp","panel_temp","voltage","current","power","efficiency","fault"
    ])

# --- Helper Functions ---
def simulate_sensor_data():
    irradiance = np.random.randint(600, 1000)
    panel_temp = np.random.uniform(30, 45)
    ambient_temp = np.random.uniform(20, 35)
    voltage = np.random.uniform(15, 19)
    current = np.random.uniform(0.6, 1.2)
    return irradiance, ambient_temp, panel_temp, voltage, current

def hybrid_infer(row_dict, image=None):
    feat = ['irradiance','panel_temp','voltage','current','power','efficiency']
    if tabular_model:
        x = np.array([[row_dict[f] for f in feat]])
        prob = tabular_model.predict_proba(x).max() if hasattr(tabular_model, "predict_proba") else None
        pred = tabular_model.predict(x)[0]
        if prob and prob < 0.7 and cnn_model and image is not None:
            arr = np.array(image.resize((64,64)).convert("L"))[...,np.newaxis]/255.0
            ip = cnn_model.predict(np.expand_dims(arr, 0)).argmax(axis=1)[0]
            return {"source":"CNN","pred":ip}
        return {"source":"Tabular","pred":pred,"conf":float(prob) if prob else None}
    else:
        # simple rule-based fallback
        if row_dict["voltage"]<voltage_min or row_dict["current"]<current_min: return {"pred":"Warning"}
        return {"pred":"Normal"}

# --- Layout ---
st.title("☀️ Solar Panel Fault Detection & Monitoring — Professional Dashboard")
tabs = st.tabs(["Live Metrics", "Historical Trends", "Thermal Analysis", "Alerts & Logs"])

# --- Tab 1: Live Metrics ---
with tabs[0]:
    st.subheader("🌞 Real-Time Sensor Input")
    left, right = st.columns([1,1])

    with left:
        mode = st.radio("Input Mode:", ["Manual Reading","CSV Upload"], index=0)
        if mode=="Manual Reading":
            irradiance = st.number_input("Irradiance (W/m²)", 800.0)
            ambient_temp = st.number_input("Ambient Temp (°C)", 30.0)
            panel_temp = st.number_input("Panel Temp (°C)", 35.0)
            voltage = st.number_input("Voltage (V)", 18.5)
            current = st.number_input("Current (A)", 0.85)
            if st.button("Predict"):
                row = {
                    "irradiance": irradiance,
                    "ambient_temp": ambient_temp,
                    "panel_temp": panel_temp,
                    "voltage": voltage,
                    "current": current
                }
                row["power"] = row["voltage"]*row["current"]
                row["efficiency"] = row["power"]/(row["irradiance"]+1e-6)
                res = hybrid_infer(row)
                st.success(f"Detected Fault: {res}")

        else:
            up = st.file_uploader("Upload CSV with required columns", type=["csv"])
            if up is not None:
                df = pd.read_csv(up)
                st.dataframe(df.head())
                if st.button("Batch Predict"):
                    df['power'] = df['voltage']*df['current']
                    df['efficiency'] = df['power']/(df['irradiance']+1e-6)
                    df['fault'] = df.apply(lambda r: hybrid_infer(r)['pred'], axis=1)
                    st.dataframe(df.head())

    with right:
        st.subheader("📊 Live Dashboard")
        if live_mode:
            placeholder = st.empty()
            for _ in range(10):
                irr, amb, panel, volt, curr = simulate_sensor_data()
                power = volt*curr
                eff = power/(irr+1e-6)
                fault = "Normal"
                row = {"irradiance":irr,"ambient_temp":amb,"panel_temp":panel,"voltage":volt,"current":curr,"power":power,"efficiency":eff}
                res = hybrid_infer(row)
                if "pred" in res: fault=res["pred"]
                st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([{**row,"fault":fault,"timestamp":datetime.now()}])], ignore_index=True)
                fig = px.line(st.session_state.data, x="timestamp", y=["voltage","current","power","efficiency"], markers=True)
                placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(1)
        else:
            st.info("Enable 'Live Sensor Data' in sidebar to simulate or connect Arduino/ESP32.")

# --- Tab 2: Historical Trends ---
with tabs[1]:
    st.subheader("📈 Historical Trends & Efficiency")
    if not st.session_state.data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.data['timestamp'], y=st.session_state.data['efficiency'], mode='lines+markers', name='Efficiency'))
        fig.add_trace(go.Scatter(x=st.session_state.data['timestamp'], y=st.session_state.data['power'], mode='lines+markers', name='Power (W)'))
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No data available yet.")

# --- Tab 3: Thermal Analysis ---
with tabs[2]:
    st.subheader("🌡 Thermal / IR Image Prediction")
    img_up = st.file_uploader("Upload Thermal Image", type=["png","jpg","jpeg"])
    if img_up:
        image = Image.open(img_up)
        st.image(image, caption="Uploaded Thermal Image")
        if cnn_model:
            arr = np.array(image.resize((64,64)).convert("L"))[...,np.newaxis]/255.0
            pred = cnn_model.predict(np.expand_dims(arr,0)).argmax(axis=1)[0]
            st.success(f"Thermal Image Prediction: {pred}")

# --- Tab 4: Alerts & Logs ---
with tabs[3]:
    st.subheader("⚠️ Alerts & Fault Logs")
    if not st.session_state.data.empty:
        alerts = st.session_state.data[st.session_state.data['fault']!="










































