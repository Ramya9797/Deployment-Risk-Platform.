import json

class Metadata:

    def create(self, train, test):

        metadata = {

            "train_rows": len(train),

            "test_rows": len(test),

            "features": len(train.columns)-1,

            "target": "target"

        }

        with open(
            "artifacts/metadata.json",
            "w"
        ) as f:

            json.dump(metadata, f, indent=4)