import sys

from src.entity.config_entity import DataIngestionConfig,DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.exception import CustomerException
from src.components.data_validation import DataValidation

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Starting TrainPipeline.start_data_ingestion") 

        try:
            data_ingestion = DataIngestion(data_ingestion_config= self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")

            logging.info("Exiting TrainPipeline.start_data_ingestion")

            return data_ingestion_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("Starting TrainPipeline.start_data_validation")
        try:
            data_validation = DataValidation(
                data_ingestion_artifact= data_ingestion_artifact,
                data_validation_config= self.data_validation_config
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Exiting TrainPipeline.start_data_validation")
        except Exception as e:
            raise CustomerException(e, sys) from e

    def run_pipeline(self) -> None :
        logging.info("Starting TrainPipeline.run_pipeline")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact= data_ingestion_artifact
            )
            
        except Exception as e:
            raise CustomerException(e, sys) from e
        