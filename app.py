from fastapi import FastAPI
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI(title="Churn Prediction API")

# Load model
artifact = joblib.load("model/random_forest_churn_model.pkl")

model = artifact["model"]
features = artifact["features"]


@app.get("/")
def home():
    return {
        "message": "Churn Prediction API is running"
    }


@app.post("/predict")
def predict(data: dict):

    # Convert input JSON to DataFrame
    input_df = pd.DataFrame([data])

    # Ensure correct feature order
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return {
        "Prediction": int(prediction),
        "Probability": float(probability)
    }