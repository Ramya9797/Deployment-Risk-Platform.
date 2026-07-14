from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

class Encoder:

    def fit_transform(self, train_df, test_df):

        categorical = [

            "environment",

            "application"

        ]

        transformer = ColumnTransformer(

            transformers=[

                (

                    "cat",

                    OneHotEncoder(

                        handle_unknown="ignore"

                    ),

                    categorical

                )

            ],

            remainder="passthrough"

        )

        train = transformer.fit_transform(train_df)

        test = transformer.transform(test_df)

        joblib.dump(

            transformer,

            "artifacts/preprocessor.pkl"

        )

        return train, test