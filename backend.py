# backend.py - FastAPI backend for Solar Fault Detection

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os
import nest_asyncio
from pyngrok import ngrok

# ----------------- Config -----------------
MODEL_DIR = "models"
TABULAR_MODEL_PATH = os.path.join(MODEL_DIR, "tabular_rf.pkl")
CNN_MODEL_PATH     = os.path.join(MODEL_DIR, "cnn_thermal.h5")

# ----------------- Load Models -----------------
tabular_model = joblib.load(TABULAR_MODEL_PATH) if os.path.exists(TABULAR_MODEL_PATH) else None
cnn_model = load_model(CNN_MODEL_PATH) if os.path.exists(CNN_MODEL_PATH) else None

# ----------------- FastAPI App -----------------
app = FastAPI(title="Solar Fault Detection API")

class SensorData(BaseModel):
    irradiance_wpm2: float
    ambient_temp_c: float
    panel_temp_c: float
    voltage_v: float
    current_a: float

FEATURES = ['irradiance_wpm2','panel_temp_c','voltage_v','current_a',
            'power_w','efficiency','temp_delta','residual_power']

def compute_features(row):
    row["power_w"] = row["voltage_v"] * row["current_a"]
    row["efficiency"] = row["power_w"] / (row["irradiance_wpm2"] + 1e-6)
    row["temp_delta"] = row["panel_temp_c"] - row["ambient_temp_c"]
    row["residual_power"] = 0.0  # single reading
    return row

@app.post("/infer")
def infer(data: SensorData):
    row = data.dict()
    row = compute_features(row)
    
    if tabular_model:
        x = np.array([[row[f] for f in FEATURES]])
        pred = tabular_model.predict(x)[0]
        conf = float(tabular_model.predict_proba(x).max() if hasattr(tabular_model,"predict_proba") else 0.0)
    else:
        pred, conf = "Model not loaded", 0.0

    return {
        "prediction": pred,
        "confidence": conf
    }

# ----------------- Run with ngrok (for Colab) -----------------
if __name__ == "__main__":
    import uvicorn
    nest_asyncio.apply()
    
    # Set your ngrok auth token
    NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    # Open public URL
    public_url = ngrok.connect(8000)
    print("Public FastAPI URL:", public_url)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
