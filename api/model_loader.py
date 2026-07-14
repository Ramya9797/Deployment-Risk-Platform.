import joblib

_model = None

def get_model():
    global _model

    if _model is None:
        _model = joblib.load("models/best_model.pkl")

    return _model


def get_model_version():
    return "1.0"