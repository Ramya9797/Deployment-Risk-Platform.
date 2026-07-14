import pandas as pd

class SaveProcessed:

    def save(

        self,

        train,

        test

    ):

        pd.DataFrame(train).to_csv(

            "data/processed/train_processed.csv",

            index=False

        )

        pd.DataFrame(test).to_csv(

            "data/processed/test_processed.csv",

            index=False

        )