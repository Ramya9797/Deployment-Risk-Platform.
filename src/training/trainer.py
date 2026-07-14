from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ModelTrainer:

    def get_models(self):

        return {

            "RandomForest":

                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                ),

            "XGBoost":

                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss"
                ),

            "LightGBM":

                LGBMClassifier(
                    random_state=42
                )
        }