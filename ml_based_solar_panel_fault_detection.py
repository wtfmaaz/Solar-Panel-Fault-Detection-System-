import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import paho.mqtt.client as mqtt
from keras.models import load_model
from prophet import Prophet

# =====================================================
# PATHS & CONSTANTS
# =====================================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_rf.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
DB_PATH = "logs/solar_faults.db"
CSV_FILE = "logs/realtime_data.csv"

# =====================================================
# DATABASE INITIALIZATION
# =====================================================
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

init_db()

# =====================================================
# EMAIL ALERT FUNCTION
# =====================================================
def send_email_alert(panel_no, fault_type):
    sender_email = "youremail@example.com"
    receiver_email = "receiver@example.com"
    msg = MIMEText(f"⚠️ Fault detected!\nPanel {panel_no} has {fault_type} fault.")
    msg['Subject'] = f"Solar Panel Fault Alert - Panel {panel_no}"
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login(sender_email, "yourpassword")
            server.send_message(msg)
        st.toast(f"📧 Email sent for Panel {panel_no}", icon="✉️")
    except Exception as e:
        st.warning(f"Email failed: {e}")

# =====================================================
# SAVE READING TO CSV
# =====================================================
def save_reading(df_row):
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=df_row.columns)
        df.to_csv(CSV_FILE, index=False)
    try:
        existing_df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        existing_df = pd.DataFrame(columns=df_row.columns)

    updated_df = pd.concat([existing_df, df_row], ignore_index=True)
    updated_df.to_csv(CSV_FILE, index=False)

# =====================================================
# MODEL LOADING
# =====================================================
@st.cache_resource
def load_tabular_model():
    if not os.path.isfile(TABULAR_MODEL_PATH):
        st.warning("⚠️ Tabular model missing!")
        return None
    return joblib.load(TABULAR_MODEL_PATH)

@st.cache_resource
def load_lstm_model():
    if not os.path.isfile(LSTM_MODEL_PATH):
        st.warning("⚠️ LSTM model missing!")
        return None
    try:
        return load_model(LSTM_MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"LSTM load error: {e}")
        return None

tabular_model = load_tabular_model()
lstm_model = load_lstm_model()

# =====================================================
# FAULT LOGGING
# =====================================================
def log_fault(panel_no, fault_type):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO fault_logs (timestamp, panel_no, fault_type) VALUES (?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), panel_no, fault_type))
    conn.commit()
    conn.close()

# =====================================================
# REAL-TIME MQTT DATA HANDLER
# =====================================================
st.session_state.setdefault("realtime_data", pd.DataFrame())

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode().strip()
        # Expected format: panel_no,voltage,current,irradiance,temp
        panel_no, voltage, current, irradiance, temp = map(float, payload.split(","))
        power = voltage * current
        efficiency = (power / irradiance) * 100 if irradiance > 0 else 0

        df_row = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "panel_no": panel_no,
            "voltage_v": voltage,
            "current_a": current,
            "irradiance_wpm2": irradiance,
            "panel_temp_c": temp,
            "power_w": power,
            "efficiency": efficiency
        }])

        # Save locally & store in memory
        save_reading(df_row)
        st.session_state.realtime_data = pd.concat(
            [st.session_state.realtime_data, df_row], ignore_index=True
        )

        # Auto fault detection
        if tabular_model is not None:
            feat_cols = ["voltage_v", "current_a", "irradiance_wpm2", "panel_temp_c", "power_w", "efficiency"]
            prediction = tabular_model.predict(df_row[feat_cols])[0]
            if prediction != 0:
                log_fault(int(panel_no), str(prediction))
                send_email_alert(int(panel_no), str(prediction))
                st.error(f"⚠️ Fault Detected on Panel {panel_no}: Type {prediction}")
    except Exception as e:
        st.warning(f"MQTT error: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
try:
    mqtt_client.connect("broker.hivemq.com", 1883, 60)
    mqtt_client.subscribe("solar/panels")
    mqtt_client.loop_start()
except Exception as e:
    st.warning(f"⚠️ MQTT broker connection failed: {e}")

# =====================================================
# STREAMLIT DASHBOARD UI
# =====================================================
st.title("🌞 Autonomous Solar Panel Monitoring & Fault Detection")

if st.session_state.realtime_data.shape[0] > 0:
    df = st.session_state.realtime_data.tail(50)

    # LIVE CHART
    st.subheader("📊 Live Sensor Readings")
    fig = px.line(df, x="timestamp", y=["voltage_v", "current_a", "irradiance_wpm2", "panel_temp_c"],
                  title="Real-Time Sensor Data (Last 50 readings)")
    st.plotly_chart(fig, use_container_width=True)

    # LSTM VOLTAGE FORECAST
    if lstm_model is not None and len(df) >= 10:
        latest_seq = df[["voltage_v", "current_a", "irradiance_wpm2", "panel_temp_c"]].tail(10).values
        latest_seq = np.expand_dims(latest_seq, axis=0)
        predicted_voltage = lstm_model.predict(latest_seq)[0][0]
        st.metric("🔮 Predicted Next Voltage", f"{predicted_voltage:.2f} V")

    # PROPHET FORECAST
    if len(df) > 10:
        try:
            df_prophet = df.copy()
            df_prophet["ds"] = pd.to_datetime(df_prophet["timestamp"])
            df_prophet["y"] = df_prophet["voltage_v"]
            prophet = Prophet()
            prophet.fit(df_prophet[["ds", "y"]])
            future = prophet.make_future_dataframe(periods=10)
            forecast = prophet.predict(future)
            fig2 = px.line(forecast, x="ds", y="yhat", title="Predicted Voltage Trend (Prophet)")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Prophet forecast error: {e}")

else:
    st.info("Waiting for real-time data from ESP32 MQTT topic...")

# =====================================================
# FAULT LOG VIEW
# =====================================================
st.subheader("🧾 Fault History Log")
conn = sqlite3.connect(DB_PATH)
logs = pd.read_sql("SELECT * FROM fault_logs ORDER BY timestamp DESC", conn)
st.dataframe(logs)
conn.close()
