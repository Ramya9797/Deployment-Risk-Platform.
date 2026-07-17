import mlflow
import mlflow.sklearn

mlflow.set_experiment("deployment-risk")

with mlflow.start_run():

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="DeploymentRiskModel"
    )