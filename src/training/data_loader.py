import joblib
import pandas as pd


class TrainingDataLoader:

    def load(self):

        X_train = joblib.load("data/processed/X_train.pkl")
        X_test = joblib.load("data/processed/X_test.pkl")

        y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

        return X_train, X_test, y_train, y_test