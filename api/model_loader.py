import mlflow.pyfunc

mlflow.set_tracking_uri("http://127.0.0.1:5000")


def get_model():

    model = mlflow.pyfunc.load_model(
        "models:/deployment-risk-model@production"
    )

    return model


def get_model_version():
    return "Production"