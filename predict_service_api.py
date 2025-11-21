# -------------------- IMPORTS --------------------
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# -------------------- LOAD TRAINED MODEL --------------------
MODEL_PATH = "trained_xgb_model.joblib"
model = joblib.load(MODEL_PATH)

# -------------------- FEATURES REQUIRED BY THE MODEL --------------------
final_features = [
    'Hour',
    'total_received',
    'mean_value_received',
    'time_diff_first_last_received',
    'total_tx_sent',
    'total_tx_sent_unique',
    'has_activity'
]

# -------------------- FASTAPI APP --------------------
app = FastAPI(title="Ethereum Transaction Fraud Detection API")

# -------------------- PREDICTION ENDPOINT --------------------
@app.post("/predict")
async def predict(request: Request):
    """
    Input JSON example:
    {
        "tx_hash": "0xa53603...",
        "Hour": 14,
        "total_received": 0.0,
        "mean_value_received": 0.0,
        "time_diff_first_last_received": 0.0,
        "total_tx_sent": 3,
        "total_tx_sent_unique": 2,
        "has_activity": 1
    }
    """
    data = await request.json()

    # Extract tx_hash (metadata only)
    tx_hash = data.pop("tx_hash", None)

    # Validate required features
    missing_features = [f for f in final_features if f not in data]
    if missing_features:
        return JSONResponse(
            status_code=400,
            content={"error": f"Missing required features: {missing_features}"}
        )

    # Prepare DataFrame for model
    df = pd.DataFrame([data], columns=final_features)

    # Make prediction
    prob = model.predict_proba(df)[0, 1]  # fraud probability
    is_fraud = bool(prob >= 0.5)          # binary prediction

    # Human-readable risk classification
    if prob >= 0.85:
        classification = "High Risk"
        message = "Transaction shows very strong fraud indicators. Immediate action recommended."
    elif prob >= 0.55:
        classification = "Medium Risk"
        message = "Transaction shows moderate suspicious activity. Monitor closely."
    else:
        classification = "Low Risk"
        message = "Transaction appears normal. No strong fraud signals detected."

    # Build response
    result = {
        "tx_hash": tx_hash,
        "fraud_probability": float(prob),
        "is_fraud": is_fraud,
        "classification": classification,
        "message": message
    }

    return JSONResponse(content=result)

# -------------------- DEV SERVER --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
