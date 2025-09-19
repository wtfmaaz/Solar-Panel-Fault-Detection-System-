import os
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import streamlit as st
import plotly.express as px
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import paho.mqtt.client as mqtt
from prophet import Prophet

# -------------------------------
# Paths & Constants
# -------------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_rf.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
DB_PATH = "logs/solar_faults.db"
os.makedirs("logs", exist_ok=True)

# -------------------------------
# Email Notification Function
# -------------------------------
def send_email_alert(panel_no, fault_type):
    sender_email = "youremail@example.com"
    receiver_email = "receiver@example.com"
    msg = MIMEText(f"Fault detected!\nPanel {panel_no} has {fault_type} fault.")
    msg['Subject'] = f"Solar Panel Fault Alert - Panel {panel_no}"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login(sender_email, "yourpassword")
            server.send_message(msg)
        st.info(f"Email sent for Panel {panel_no}")
    except Exception as e:
        st.warning(f"Email failed: {e}")

# -------------------------------
# SQLite Logging
# -------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS fault_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            panel_no INTEGER,
            fault_type TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_fault(panel_no, fault_type):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO fault_logs (timestamp, panel_no, fault_type) VALUES (?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), panel_no, fault_type))
    conn.commit()
    conn.close()

# -------------------------------
# Model Loading (Safe)
# -------------------------------
@st.cache_resource
def load_tabular_model():
    if not os.path.isfile(TABULAR_MODEL_PATH):
        st.warning(f"Tabular model not found at {TABULAR_MODEL_PATH}. Upload or train one.")
        return None
    return joblib.load(TABULAR_MODEL_PATH)

@st.cache_resource
def load_lstm_model():
    if not os.path.isfile(LSTM_MODEL_PATH):
        st.warning(f"LSTM model not found at {LSTM_MODEL_PATH}. Upload or train one.")
        return None
    return load_model(LSTM_MODEL_PATH)

tabular_model = load_tabular_model()
lstm_model = load_lstm_model()

# -------------------------------
# MQTT Real-time Data
# -------------------------------
st.session_state.setdefault("realtime_data", pd.DataFrame())

def on_message(client, userdata, msg):
    # Expected format: panel_no,voltage,current,irradiance,temp
    try:
        payload = msg.payload.decode()
        panel_no, voltage, current, irradiance, temp = map(float, payload.split(","))
        df = pd.DataFrame({
            "panel_no": [panel_no],
            "voltage": [voltage],
            "current": [current],
            "irradiance": [irradiance],
            "temp": [temp]
        })
        st.session_state.realtime_data = pd.concat([st.session_state.realtime_data, df], ignore_index=True)
    except Exception as e:
        st.error(f"MQTT parse error: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
try:
    mqtt_client.connect("broker.hivemq.com", 1883, 60)
    mqtt_client.subscribe("solar/panels")
    mqtt_client.loop_start()
except Exception as e:
    st.warning(f"MQTT broker not connected: {e}")

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("☀️ Advanced Solar Panel Fault Detection & Monitoring")

input_mode = st.radio("Input mode:", ["Manual Reading", "CSV Upload"])

if input_mode == "Manual Reading":
    panel_no = st.number_input("Panel Number", 1, 100, 1)
    voltage = st.number_input("Voltage (V)", 0.0, 50.0, 18.5)
    current = st.number_input("Current (A)", 0.0, 20.0, 0.85)
    irradiance = st.number_input("Irradiance (W/m²)", 0.0, 1200.0, 800.0)
    temp = st.number_input("Panel Temperature (°C)", -20.0, 100.0, 35.0)
    
    manual_input = pd.DataFrame({
        "panel_no": [panel_no],
        "voltage": [voltage],
        "current": [current],
        "irradiance": [irradiance],
        "temp": [temp]
    })
else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        manual_input = pd.read_csv(uploaded_file)

# -------------------------------
# Fault Prediction
# -------------------------------
def predict_fault(df):
    if tabular_model:
        features = df[["voltage", "current", "irradiance", "temp"]]
        df["fault"] = tabular_model.predict(features)
        for _, row in df.iterrows():
            if row["fault]()

































