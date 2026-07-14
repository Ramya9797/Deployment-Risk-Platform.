from mlflow import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name="deployment-risk-model",
    version=1,
    stage="Staging"
)