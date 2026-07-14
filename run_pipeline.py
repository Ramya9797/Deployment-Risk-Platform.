from src.ingestion.pipeline import IngestionPipeline
from src.feature_engineering.pipeline import FeatureEngineeringPipeline
from src.training.pipeline import TrainingPipeline

if __name__ == "__main__":

    IngestionPipeline().run()

    FeatureEngineeringPipeline().run()

    TrainingPipeline().run()