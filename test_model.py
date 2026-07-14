import mlflow
import mlflow.pyfunc
from sklearn.datasets import load_breast_cancer

# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the model from the registry using the staging alias
model = mlflow.pyfunc.load_model(
    "models:/deployment-risk-model@staging"
)

# Load sample data
data = load_breast_cancer()
X = data.data

# Predict the first five rows
predictions = model.predict(X[:5])

print("Actual labels:")
print(data.target[:5])

print("Predictions:")
print(predictions)