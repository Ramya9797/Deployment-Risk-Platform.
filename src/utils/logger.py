import logging
import os

os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    filename="artifacts/ingestion.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)