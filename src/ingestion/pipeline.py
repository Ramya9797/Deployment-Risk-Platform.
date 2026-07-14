from src.ingestion.data_loader import DataLoader
from src.ingestion.validator import Validator
from src.ingestion.splitter import Splitter
from src.ingestion.metadata import Metadata
from src.utils.logger import logger

class IngestionPipeline:

    def run(self):

        logger.info("Loading data")

        loader = DataLoader(
            "data/raw/deployment_history.csv"
        )

        df = loader.load()

        logger.info("Validating")

        Validator(
            df,
            "config/schema.yaml"
        ).validate()

        logger.info("Splitting")

        train, test = Splitter().split(df)

        logger.info("Creating metadata")

        Metadata().create(train, test)

        logger.info("Pipeline completed")