import os
from dataclasses import dataclass
from datetime import datetime

from src.constant.training_pipeline import PIPELINE_NAME,ARTIFACT_DIR, DATA_INGESTION_COLLECTION_NAME,DATA_INGESTION_DIR_NAME,DATA_INGESTION_FEATURE_STORE_DIR,DATA_INGESTION_INGESTED_DIR,FILE_NAME,TRAIN_FILE_NAME,TEST_FILE_NAME,DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(PIPELINE_NAME, ARTIFACT_DIR,TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config :TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
    ingested_data_dir: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR)
    training_file_path: str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME