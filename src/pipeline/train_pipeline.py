import sys

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.exception import CustomerException

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Starting TrainPipeline.start_data_ingestion") 

        try:
            data_ingestion = DataIngestion(data_ingestion_config= self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")

            logging.info("Exiting TrainPipeline.start_data_ingestion")

            return data_ingestion_artifact
        except Exception as e:
            raise CustomerException(e, sys)

    def run_pipeline(self) -> None :
        logging.info("Starting TrainPipeline.run_pipeline")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise CustomerException(e, sys)
        