import sys

from src.entity.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig
from src.cloud_storage.aws_storage import S3BucketOperations
from src.ml.model.s3_estimator import CustomerClusterEstimator
from src.exception import CustomerException
from src.logger import logging

class ModelPusher:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_pusher_config: ModelPusherConfig
    ):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        self.s3 = S3BucketOperations()
        self.src_estimator= CustomerClusterEstimator(
            bucket_name= self.model_pusher_config.bucket_name,
            model_path= self.model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Starting ModelPusher.initiate_model_pusher")
            self.src_estimator.save_model(
                from_file = self.model_trainer_artifact.trained_model_file_path
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name= self.model_pusher_config.bucket_name,
                s3_model_path= self.model_pusher_config.s3_model_key_path
            )

            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            return model_pusher_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e