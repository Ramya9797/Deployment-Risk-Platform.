import pandas as pd

class FeatureCreator:

    def create(self, df):

        df = df.copy()

        df["risk_index"] = (
            df["changed_files"]
            * (df["failed_tests"] + 1)
        )

        df["deployment_complexity"] = (

            df["changed_files"]

            + df["database_changes"] * 30

            + df["infra_changes"] * 20

        )

        df["coverage_gap"] = 100 - df["test_coverage"]

        df["experience_score"] = (

            df["developer_experience"]

            / (df["changed_files"] + 1)

        )

        return df