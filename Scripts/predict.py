"""
Predict script for fraud detection model
----------------------------------------
Load the trained model and run predictions on new transaction data.
"""

import argparse
import pandas as pd
import joblib
import numpy as np

parser = argparse.ArgumentParser(description="Fraud Prediction Script")
parser.add_argument("--model-path", type=str, default="./models/fraud_model.pkl", help="Path to trained model")
parser.add_argument("--input-file", type=str, required=True, help="Path to CSV with new transaction data")
args = parser.parse_args()

# Load model
model = joblib.load(args.model_path)
print(f"[INFO] Loaded model from {args.model_path}")

# Load new data
data = pd.read_csv(args.input_file)
X = data.values

# Predict
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]

data['fraud_prediction'] = preds
data['fraud_probability'] = probs
data.to_csv("predictions.csv", index=False)

print("[INFO] Predictions saved to predictions.csv")
