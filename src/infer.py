from fastapi import FastAPI, Query, Response, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import json

MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")

app = FastAPI(title="Churn Prediction API", version="1.0.0")

class Payload(BaseModel):
    features: dict

def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model not found. Train first to create artifacts/model.joblib"
        )
    return joblib.load(MODEL_PATH)

model = None

@app.on_event("startup")
def startup():
    global model
    model = load_model()

@app.get("/")
def health():
    return {"status": "ok", "message": "Churn API running"}

@app.post("/predict")
def predict(payload: Payload, pretty: int = Query(0, description="Set 1 to pretty-print")):
    # Convert incoming features -> DataFrame with a single row
    X = pd.DataFrame([payload.features])

    # Predict churn probability (assumes sklearn pipeline with predict_proba)
    p = float(model.predict_proba(X)[0][1])
    label = int(p > 0.5)

    body = {
        "input_features": payload.features,
        "churn_probability": round(p, 4),
        "prediction": "Yes" if label == 1 else "No",
        "threshold_used": 0.5,
        "model_info": {
            "algorithm": "GradientBoostingClassifier",
            "version": "1.0.0"
        }
    }

    if pretty == 1:
        return Response(content=json.dumps(body, indent=2), media_type="application/json")
    return body

@app.get("/metrics")
def get_metrics(pretty: int = Query(0, description="Set 1 to pretty-print")):
    """
    Returns training/evaluation metrics saved at artifacts/metrics.json
    Example content:
    {
      "roc_auc": 0.84,
      "accuracy": 0.80,
      "f1": 0.71
    }
    """
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found. Train the model first.")
    try:
        metrics = json.loads(METRICS_PATH.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {e}")

    if pretty == 1:
        return Response(content=json.dumps(metrics, indent=2), media_type="application/json")
    return metrics
