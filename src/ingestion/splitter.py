from sklearn.model_selection import train_test_split

class Splitter:

    def split(self, df):

        train, test = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        train.to_csv(
            "data/train/train.csv",
            index=False
        )

        test.to_csv(
            "data/test/test.csv",
            index=False
        )

        return train, test