import os
import time
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --------------------------------------------------
# Optional MLflow configuration
# --------------------------------------------------
USE_MLFLOW = False

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Default")
    USE_MLFLOW = True

# --------------------------------------------------
# Dataset
# --------------------------------------------------
data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
)

os.makedirs("models", exist_ok=True)

best_model = None
best_accuracy = 0

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100,
        random_state=42,
    ),
}

for name, model in models.items():

    if USE_MLFLOW:
        run = mlflow.start_run(run_name=name)
    else:
        run = None

    start = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    if USE_MLFLOW:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("training_time", train_time)

        mlflow.log_params(model.get_params())

        if name == "RandomForest":
            mlflow.sklearn.log_model(model, artifact_path="model")

        elif name == "XGBoost":
            mlflow.xgboost.log_model(model, artifact_path="model")

        elif name == "LightGBM":
            mlflow.lightgbm.log_model(model, artifact_path="model")

        mlflow.end_run()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

joblib.dump(best_model, "models/best_model.pkl")

print("Best model saved to models/best_model.pkl")
print("Best Accuracy:", best_accuracy)