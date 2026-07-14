import pandas as pd

from src.feature_engineering.cleaner import DataCleaner
from src.feature_engineering.feature_creator import FeatureCreator
from src.feature_engineering.encoder import Encoder
from src.feature_engineering.scaler import FeatureScaler
from src.feature_engineering.preprocessing import SaveProcessed


class FeatureEngineeringPipeline:

    def run(self):

        train = pd.read_csv(

            "data/train/train.csv"

        )

        test = pd.read_csv(

            "data/test/test.csv"

        )

        cleaner = DataCleaner()

        train = cleaner.clean(train)

        test = cleaner.clean(test)

        creator = FeatureCreator()

        train = creator.create(train)

        test = creator.create(test)

        y_train = train["target"]

        y_test = test["target"]

        X_train = train.drop("target", axis=1)

        X_test = test.drop("target", axis=1)

        encoder = Encoder()

        X_train, X_test = encoder.fit_transform(

            X_train,

            X_test

        )

        scaler = FeatureScaler()

        X_train, X_test = scaler.scale(

            X_train,

            X_test

        )

        SaveProcessed().save(

            X_train,

            X_test

        )

        print("Feature Engineering Completed")