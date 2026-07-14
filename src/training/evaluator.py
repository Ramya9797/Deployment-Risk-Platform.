from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class Evaluator:

    def evaluate(

        self,

        model,

        X_test,

        y_test

    ):

        prediction = model.predict(X_test)

        probability = model.predict_proba(X_test)[:,1]

        return {

            "accuracy":

                accuracy_score(y_test,prediction),

            "precision":

                precision_score(y_test,prediction),

            "recall":

                recall_score(y_test,prediction),

            "f1":

                f1_score(y_test,prediction),

            "auc":

                roc_auc_score(y_test,probability)

        }