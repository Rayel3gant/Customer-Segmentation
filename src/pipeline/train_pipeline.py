import sys

from src.entity.config_entity import DataIngestionConfig,DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact , DataTransformationArtifact, ModelTrainerArtifact
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.exception import CustomerException
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

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
            return data_validation_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e
        
    def start_data_transformation(
        self,
        data_ingestion_artifact = DataIngestionArtifact,
        data_validation_artifact = DataValidationArtifact
    ) -> DataTransformationArtifact :
        try:
            logging.info("Starting TrainPipeline.start_data_transformation")

            data_transformation = DataTransformation(
                data_validation_artifact= data_validation_artifact,
                data_ingestion_artifact= data_ingestion_artifact,
                data_transformation_config= self.data_transformation_config
            )

            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact : DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Starting TrainPipeline.start_model_trainer")

            model_trainer = ModelTrainer(
                data_transformation_artifact= data_transformation_artifact,
                model_trainer_config= self.model_trainer_config
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e

    def run_pipeline(self) -> None :
        logging.info("Starting TrainPipeline.run_pipeline")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact= data_ingestion_artifact
            )

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact= data_ingestion_artifact,
                data_validation_artifact= data_validation_artifact
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact= data_transformation_artifact
            )
            logging.info("Model Trained Successfully.")
        except Exception as e:
            raise CustomerException(e, sys) from e
        