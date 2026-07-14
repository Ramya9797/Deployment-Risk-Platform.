import os
import joblib

_model = None


def get_model():
    global _model

    if _model is None:

        model_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                "best_model.pkl"
            )
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found:\n{model_path}\n"
                "Run train.py first."
            )

        _model = joblib.load(model_path)

    return _model


def get_model_version():
    return "Local"