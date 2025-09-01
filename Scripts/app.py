"""
FastAPI Microservice for Fraud Detection
----------------------------------------
This microservice loads the trained LightGBM fraud detection model
and exposes a REST API for prediction.

Run:
    uvicorn scripts.app:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST "http://127.0.0.1:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"transactions": [[0.1, -0.5, 1.2, ..., 234.5]]}'
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "./models/fraud_model.pkl"

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="REST API for enterprise fraud detection pipeline",
    version="1.0"
)

# ----------------------------
# Load Model
# ----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first using fraud_pipeline.py.")
model = joblib.load(MODEL_PATH)

# ----------------------------
# Request Schema
# ----------------------------
class PredictionRequest(BaseModel):
    transactions: list

class PredictionResponse(BaseModel):
    predictions: list
    probabilities: list

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    X = np.array(data.transactions)
    preds = model.predict(X).tolist()
    probs = model.predict_proba(X)[:, 1].tolist()
    return PredictionResponse(predictions=preds, probabilities=probs)
