import os
import joblib
import mlflow
import mlflow.sklearn

from src.training.data_loader import TrainingDataLoader
from src.training.trainer import ModelTrainer
from src.training.evaluator import Evaluator


class TrainingPipeline:

    def run(self):

        # Create models folder
        os.makedirs("models", exist_ok=True)

        # Use local MLflow tracking
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Deployment Risk Prediction")

        X_train, X_test, y_train, y_test = TrainingDataLoader().load()

        models = ModelTrainer().get_models()

        best_model = None
        best_score = -1

        for name, model in models.items():

            with mlflow.start_run(run_name=name):

                model.fit(X_train, y_train)

                metrics = Evaluator().evaluate(
                    model,
                    X_test,
                    y_test
                )

                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model"
                )

                if metrics["f1"] > best_score:
                    best_score = metrics["f1"]
                    best_model = model

        # Save best model locally
        joblib.dump(best_model, "models/best_model.pkl")

        print(f"Training completed.")
        print(f"Best F1 Score: {best_score:.4f}")
        print("Best model saved to models/best_model.pkl")


if __name__ == "__main__":
    TrainingPipeline().run()