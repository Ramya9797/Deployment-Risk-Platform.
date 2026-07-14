import yaml
import json

class Validator:

    def __init__(self, df, schema_path):

        self.df = df

        self.schema_path = schema_path

    def validate(self):

        report = {}

        with open(self.schema_path) as f:

            schema = yaml.safe_load(f)

        expected = schema["columns"]

        report["missing_columns"] = []

        report["wrong_dtype"] = []

        for column, dtype in expected.items():

            if column not in self.df.columns:

                report["missing_columns"].append(column)

            else:

                actual = str(self.df[column].dtype)

                if dtype == "int":

                    if "int" not in actual:

                        report["wrong_dtype"].append(column)

                elif dtype != actual:

                    report["wrong_dtype"].append(column)

        report["missing_values"] = self.df.isnull().sum().to_dict()

        report["duplicates"] = int(self.df.duplicated().sum())

        report["rows"] = len(self.df)

        report["columns"] = len(self.df.columns)

        with open("artifacts/validation_report.json", "w") as f:

            json.dump(report, f, indent=4)

        if report["missing_columns"]:

            raise Exception("Schema validation failed")

        return report