from api.model_loader import get_model

model = get_model()


def predict(features):

    result = model.predict([features])

    prediction = int(result[0])

    # Temporary fallback if your model only returns prediction
    if prediction == 1:
        label = "Benign"
        risk_score = 0.0
    else:
        label = "Malignant"
        risk_score = 1.0

    return prediction, label, risk_score