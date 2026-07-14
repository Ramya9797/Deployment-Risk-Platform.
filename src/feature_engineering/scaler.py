from sklearn.preprocessing import StandardScaler
import joblib

class FeatureScaler:

    def scale(self, train, test):

        scaler = StandardScaler(with_mean=False)

        train_scaled = scaler.fit_transform(train)

        test_scaled = scaler.transform(test)

        joblib.dump(

            scaler,

            "artifacts/scaler.pkl"

        )

        return train_scaled, test_scaled