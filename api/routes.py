from fastapi import APIRouter
from api.schemas import PredictionRequest
from api.predictor import predict
import uuid
import time

from api.metrics import (
    prediction_requests_total,
    failed_predictions_total,
    high_risk_predictions_total,
    prediction_latency_seconds
)

router = APIRouter()

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

        return {
            "request_id": request_id,
            "prediction": int(prediction),
            "class_name": label,
            "risk_score": round(risk_score, 4),
            "latency_ms": round(latency * 1000, 2)
        }

    except Exception:
        failed_predictions_total.inc()
        raise