from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
from api.predictor import predict
import uuid
import time

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest
)

# ---------------------------------------------------
# Connect to MLflow
# ---------------------------------------------------
app = FastAPI()

# ---------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

failed_predictions_total = Counter(
    "failed_predictions_total",
    "Total failed prediction requests"
)

high_risk_predictions_total = Counter(
    "high_risk_predictions_total",
    "Total malignant (high-risk) predictions"
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------
class PredictionRequest(BaseModel):
    features: List[float]

# ---------------------------------------------------
# Home
# ---------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Deployment Risk Model API is running"
    }

# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------
@app.post("/predict")
def make_prediction(request: PredictionRequest):

    request_id = str(uuid.uuid4())
    start = time.time()

    prediction_requests_total.inc()

    try:
        prediction, label, risk_score = predict(request.features)

        if prediction == 0:
            high_risk_predictions_total.inc()

        latency = time.time() - start

        prediction_latency_seconds.observe(latency)

        return {
            "request_id": request_id,
            "prediction": int(prediction),
            "class_name": label,
            "risk_score": round(risk_score, 4),
            "latency_ms": round(latency * 1000, 2)
        }

    except Exception as e:
        failed_predictions_total.inc()
        raise e

# ---------------------------------------------------
# Metrics Endpoint
# ---------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )