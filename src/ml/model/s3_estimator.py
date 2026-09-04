import sys
import pandas as pd

from src.cloud_storage.aws_storage import S3BucketOperations
from src.ml.model.estimator import CustomerSegmentationModel
from src.exception import CustomerException

class CustomerClusterEstimator:
    def __init__(
        self,
        bucket_name: str,
        model_path: str
    ):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = S3BucketOperations()
        self.loaded_model: CustomerSegmentationModel = None

    def is_model_present(self, model_path: str) -> bool :
        try:
            return self.s3.s3_key_path_available(
                bucket_name= self.bucket_name,
                s3_key= model_path
            )
        except Exception as e:
            raise CustomerException(e, sys) from e

    def load_model(self) -> CustomerSegmentationModel:
        return self.s3.load_model(
            bucket_name= self.bucket_name,
            model_name= self.model_path
        )

    def predict(
        self,
        dataframe: pd.DataFrame
    ):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise CustomerException(e,sys)