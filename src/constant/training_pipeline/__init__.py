import os

PIPELINE_NAME: str = "src"
ARTIFACT_DIR: str = "artifact"
LOG_DIR = "logs"
LOG_FILE = "customer_segmentation.log"

FILE_NAME: str = "customer.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

DATA_INGESTION_COLLECTION_NAME: str = ""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")