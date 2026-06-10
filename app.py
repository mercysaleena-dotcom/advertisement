from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# ==========================================
# Load Trained Model
# ==========================================
with open("advertising_model.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================================
# Create FastAPI App
# ==========================================
app = FastAPI(
    title="Advertising Sales Prediction API",
    description="Predict Sales based on TV, Radio and Newspaper Advertising Budget",
    version="1.0"
)

# ==========================================
# Input Schema
# ==========================================
class AdvertisingInput(BaseModel):
    TV: float
    radio: float
    newspaper: float
@app.get("/")
def root():
    return RedirectResponse(url="/docs")
# ==========================================
# Home Route
# ==========================================
@app.get("/")
def home():
    return {
        "message": "Advertising Sales Prediction API is Running"
    }

# ==========================================
# Prediction Route
# ==========================================
@app.post("/predict")
def predict(data: AdvertisingInput):

    input_df = pd.DataFrame({
        "TV": [data.TV],
        "radio": [data.radio],
        "newspaper": [data.newspaper]
    })

    prediction = model.predict(input_df)

    return {
        "TV": data.TV,
        "radio": data.radio,
        "newspaper": data.newspaper,
        "Predicted Sales": round(float(prediction[0]), 2)
    }

# ==========================================
# Health Check Route
# ==========================================
@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }
