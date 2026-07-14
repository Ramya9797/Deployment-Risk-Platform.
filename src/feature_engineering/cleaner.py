import pandas as pd

class DataCleaner:

    def clean(self, df):

        df = df.copy()

        df.drop_duplicates(inplace=True)

        df.reset_index(drop=True, inplace=True)

        return df