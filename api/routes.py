from fastapi import APIRouter
from fastapi.responses import FileResponse
from api.schemas import PredictionRequest
from api.predictor import predict

import uuid
import time
import os
import pandas as pd

from api.metrics import (
    prediction_requests_total,
    failed_predictions_total,
    high_risk_predictions_total,
    prediction_latency_seconds
)

router = APIRouter()

# Absolute path to monitoring folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MONITORING_DIR = os.path.join(BASE_DIR, "monitoring")
LOG_FILE = os.path.join(MONITORING_DIR, "current_data.csv")
REPORT_FILE = os.path.join(MONITORING_DIR, "evidently_report.html")


@router.get("/health")
def health():
    return {"status": "healthy"}

@router.get("/report")
def get_report():

    if not os.path.exists(REPORT_FILE):
        return {
            "message": "Evidently report not found. Run drift.py first."
        }

    return FileResponse(
        path=REPORT_FILE,
        media_type="text/html",
        filename="evidently_report.html"
    )


@router.post("/predict")
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

        # Create monitoring directory if it doesn't exist
        os.makedirs(MONITORING_DIR, exist_ok=True)

        # Prepare row for CSV
        row = {}

        for i, value in enumerate(request.features):
            row[f"feature_{i+1}"] = value

        row["prediction"] = int(prediction)
        row["class_name"] = label
        row["risk_score"] = float(risk_score)

        df = pd.DataFrame([row])

        # Debug output
        print("=" * 60)
        print("Saving prediction log...")
        print("CSV Path:", LOG_FILE)
        print(df)
        print("=" * 60)

        # Save to CSV
        if os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_FILE, index=False)

        print("Prediction log saved successfully!")

        return {
            "request_id": request_id,
            "prediction": int(prediction),
            "class_name": label,
            "risk_score": round(risk_score, 4),
            "latency_ms": round(latency * 1000, 2)
        }

    except Exception as e:
        failed_predictions_total.inc()
        print("Prediction Error:", str(e))
        raise