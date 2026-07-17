from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess


def feature_engineering():
    print("Running feature engineering...")
    # Add preprocessing if needed


def train_model():
    print("Training model...")
    subprocess.run(["python", "train.py"], check=True)


def evaluate_model():
    print("Evaluating model...")
    # Add evaluation logic here


def register_model():
    print("Registering model to MLflow...")
    # train.py already logs to MLflow
    # or call another script if needed


default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 1, 1),
}

with DAG(
    dag_id="daily_retraining_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:

    feature_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    register_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    feature_task >> train_task >> evaluate_task >> register_task