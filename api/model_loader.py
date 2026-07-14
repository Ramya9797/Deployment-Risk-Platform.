import mlflow.pyfunc


def get_model():
    model = mlflow.pyfunc.load_model("./models/model")
    return model


def get_model_version():
    return "Production"