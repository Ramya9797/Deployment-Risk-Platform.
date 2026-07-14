import os
import joblib

_model = None

def get_model():
    global _model

    if _model is None:
        model_path = os.path.join("model", "best_models.pkl")
        _model = joblib.load(model_path)

    return _model


def get_model_version():
    return "1.0"