from prometheus_client import Counter, Histogram

# Count total prediction requests
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

# Count failed predictions
failed_predictions_total = Counter(
    "failed_predictions_total",
    "Total number of failed prediction requests"
)

# Count high-risk predictions
high_risk_predictions_total = Counter(
    "high_risk_predictions_total",
    "Total number of high-risk predictions"
)

# Measure prediction latency
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)