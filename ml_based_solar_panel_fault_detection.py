# advanced_solar_monitoring_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time, json
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import paho.mqtt.client as mqtt
from pyfcm import FCMNotification

st.set_page_config(page_title="☀Advanced Solar Panel Monitoring", layout="wide")

# ----------------- CONFIG -----------------
MODEL_DIR = "models"
TABULAR_MODEL_PATH = f"{MODEL_DIR}/tabular_rf.pkl"
CNN_MODEL_PATH = f"{MODEL_DIR}/cnn_thermal.h5"
LSTM_MODEL_PATH = f"{MODEL_DIR}/lstm_efficiency.h5"
DATABASE_PATH = "panel_logs.db"
PANEL_COUNT = 20
PANELS_PER_ROW = 5

# Email and push config
EMAIL_SENDER = "youremail@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "receiver@gmail.com"
FCM_API_KEY = "YOUR_FCM_SERVER_KEY"

# MQTT config
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "solar_panel/+/data"

# ----------------- MODELS -----------------
@st.cache_resource
def load_tabular_model(): return joblib.load(TABULAR_MODEL_PATH)
@st.cache_resource
def load_cnn_model(): return load_model(CNN_MODEL_PATH)
@st.cache_resource
def load_lstm_model(): return load_model(LSTM_MODEL_PATH)

tabular_model = load_tabular_model()
cnn_model = load_cnn_model()
lstm_model = load_lstm_model()

# ----------------- DATABASE -----------------
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS panel_logs
                 (timestamp TEXT, panel_id TEXT, voltage REAL, current REAL,
                  irradiance REAL, ambient_temp REAL, panel_temp REAL,
                  power REAL, efficiency REAL, fault TEXT)''')
    conn.commit()
    return conn

conn = init_db()
cursor = conn.cursor()

# ----------------- PANEL STATUS -----------------
def init_panel_status():
    return pd.DataFrame({
        "panel_id":[f"P{i+1}" for i in range(PANEL_COUNT)],
        "row":[i//PANELS_PER_ROW for i in range(PANEL_COUNT)],
        "col":[i%PANELS_PER_ROW for i in range(PANEL_COUNT)],
        "efficiency":1.0,
        "fault":"Normal"
    })

if "panel_status" not in st.session_state: st.session_state.panel_status = init_panel_status()
if "alerts" not in st.session_state: st.session_state.alerts = []

# ----------------- EMAIL & PUSH ALERT -----------------
push_service = FCMNotification(api_key=FCM_API_KEY)

def send_email_alert(panel_id,fault,efficiency):
    try:
        msg = MIMEMultipart()
        msg["From"]=EMAIL_SENDER
        msg["To"]=EMAIL_RECEIVER
        msg["Subject"]=f"⚠️ Solar Panel Alert: {panel_id}"
        body=f"Panel {panel_id} reported {fault}.\nEfficiency: {efficiency:.2f}\nTime: {datetime.utcnow()}"
        msg.attach(MIMEText(body,"plain"))
        server=smtplib.SMTP("smtp.gmail.com",587)
        server.starttls()
        server.login(EMAIL_SENDER,EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        st.error(f"Email alert failed: {e}")

def send_push_alert(panel_id,fault,efficiency):
    try:
        message_title=f"⚠️ Panel {panel_id} Alert"
        message_body=f"{fault} detected.\nEfficiency: {efficiency:.2f}"
        push_service.notify_topic_subscribers(topic_name="solar_alerts", message_title=message_title, message_body=message_body)
    except Exception as e:
        st.error(f"Push alert failed: {e}")

# ----------------- PREDICTIVE MAINTENANCE -----------------
def prophet_predict_efficiency(df):
    df_pm=df[['timestamp','efficiency']].rename(columns={'timestamp':'ds','efficiency':'y'})
    m=Prophet()
    m.fit(df_pm)
    future=m.make_future_dataframe(periods=60,freq='T')
    forecast=m.predict(future)
    current_eff=df_pm['y'].tail(60).mean()
    pred_eff=forecast['yhat'].iloc[-1]
    return current_eff,pred_eff

def lstm_predict_efficiency(df):
    seq_length=10
    df_sorted=df.sort_values('timestamp')
    data=df_sorted['efficiency'].values
    X=[]
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
    X=np.array(X).reshape(-1,seq_length,1)
    if len(X)==0: return 1.0
    pred=lstm_model.predict(X)
    return pred[-1][0]

# ----------------- PANEL ARRAY VISUALIZATION -----------------
def plot_panel_array():
    df=st.session_state.panel_status
    color_map={"Normal":"green","Minor":"yellow","Faulty":"red"}
    fig=go.Figure()
    for _,row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row.col], y=[row.row],
            mode="markers+text",
            marker=dict(size=50,color=color_map.get(row.fault,"green"),line=dict(color="black",width=2)),
            text=row.panel_id,textposition="middle center",showlegend=False
        ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(width=600,height=400,title="Panel Array Status")
    st.plotly_chart(fig)

# ----------------- STREAMLIT UI -----------------
st.title("☀️ Advanced Solar Panel Monitoring System")

left,right=st.columns([1,1])

with left:
    st.subheader("Sensor Input / CSV Upload")
    mode=st.radio("Input mode:",["Manual","CSV Upload"])
    if mode=="Manual":
        voltage=st.number_input("Voltage (V)",18.5)
        current=st.number_input("Current (A)",0.85)
        irradiance=st.number_input("Irradiance (W/m²)",800)
        ambient_temp=st.number_input("Ambient Temp (°C)",30)
        panel_temp=st.number_input("Panel Temp (°C)",35)
        panel_id=st.selectbox("Panel ID",st.session_state.panel_status.panel_id.tolist())
        if st.button("Predict Fault"):
            power=voltage*current
            efficiency=power/(irradiance+1e-6)
            temp_delta=panel_temp-ambient_temp
            row=np.array([[irradiance,panel_temp,voltage,current,power,efficiency,temp_delta,0.0]])
            pred_fault=tabular_model.predict(row)[0]
            conf=tabular_model.predict_proba(row).max()
            status="Normal"
            if efficiency<0.85: status="Faulty"
            elif efficiency<0.95: status="Minor"
            st.session_state.panel_status.loc[st.session_state.panel_status.panel_id==panel_id,["efficiency","fault"]]=[efficiency,status]
            cursor.execute("INSERT INTO panel_logs VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (datetime.utcnow(),panel_id,voltage,current,irradiance,ambient_temp,panel_temp,power,efficiency,pred_fault))
            conn.commit()
            if status=="Faulty":
                st.session_state.alerts.append(f"Panel {panel_id} Faulty!")
                send_email_alert(panel_id,pred_fault,efficiency)
                send_push_alert(panel_id,pred_fault,efficiency)
            st.success(f"Panel {panel_id} fault: {pred_fault} ({conf:.2f})")
    else:
        up=st.file_uploader("Upload CSV", type=["csv"])
        if up:
            df_csv=pd.read_csv(up)
            st.dataframe(df_csv.head())

with right:
    st.subheader("Panel Array Visualization")
    plot_panel_array()
    st.subheader("Alerts / Notifications")
    if st.session_state.alerts:
        for a in st.session_state.alerts[-10:]:
            st.warning(a)
    else:
        st.info("No alerts yet.")
    st.subheader("Predictive Maintenance")
    df_hist=pd.read_sql("SELECT * FROM panel_logs",conn)
    if not df_hist.empty:
        current_eff, pred_eff=prophet_predict_efficiency(df_hist)
        lstm_eff=lstm_predict_efficiency(df_hist)
        st.metric("Current Efficiency (Prophet)",f"{current_eff:.2f}")
        st.metric("Predicted Efficiency (Prophet)",f"{pred_eff:.2f}")
        st.metric("Predicted Efficiency (LSTM)",f"{lstm_eff:.2f}")
        if pred_eff<current_eff*0.9: st.warning("⚠️ Predicted efficiency drop (Prophet)!")
        if lstm_eff<current_eff*0.9: st.warning("⚠️ Predicted efficiency drop (LSTM)!")

st.markdown("---")
st.info("Panel array updates in real-time from MQTT devices.")

# ----------------- MQTT STREAM -----------------
def on_message(client,userdata,msg):
    data=json.loads(msg.payload.decode())
    panel_id=data["panel_id"]
    voltage=data["voltage"]
    current=data["current"]
    irradiance=data["irradiance"]
    ambient_temp=data["ambient_temp"]
    panel_temp=data["panel_temp"]
    power=voltage*current
    efficiency=power/(irradiance+1e-6)
    status="Normal"
    if efficiency<0.85: status="Faulty"
    elif efficiency<0.95: status="Minor"
    st.session_state.panel_status.loc[st.session_state.panel_status.panel_id==panel_id,["efficiency","fault"]]=[efficiency,status]
    cursor.execute("INSERT INTO panel_logs VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (datetime.utcnow(),panel_id,voltage,current,irradiance,ambient_temp,panel_temp,power,efficiency,status))
    conn.commit()
    if status=="Faulty":
        alert_msg=f"Panel {panel_id} Faulty!"
        st.session_state.alerts.append(alert_msg)
        send_email_alert(panel_id,"Faulty",efficiency)
        send_push_alert(panel_id,"Faulty",efficiency)

client=mqtt.Client()
client.on_message=on_message
client.connect(MQTT_BROKER,MQTT_PORT,60)
client.subscribe(MQTT_TOPIC)
client.loop_start()



































