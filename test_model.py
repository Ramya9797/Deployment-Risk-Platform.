import os
import joblib
from sklearn.datasets import load_breast_cancer


MODEL_PATH = "models/best_model.pkl"


def test_model_exists():

    assert os.path.exists(MODEL_PATH)


def test_model_prediction():

    model = joblib.load(MODEL_PATH)

    data = load_breast_cancer()

    X = data.data

    predictions = model.predict(X[:5])

    print("Actual labels:")
    print(data.target[:5])

    print("Predictions:")
    print(predictions)

    assert len(predictions) == 5