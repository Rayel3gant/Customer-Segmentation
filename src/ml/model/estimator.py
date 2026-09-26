import pandas as pd
import sys

from src.logger import logging
from src.exception import CustomerException

class CustomerSegmentationModel:
    def __init__(
        self,
        preprocessing_obj: object,
        trained_model_obj: object
    ):
        self.preprocessing_obj = preprocessing_obj
        self.trained_model_obj = trained_model_obj

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_obj.transform(X)

            return self.trained_model_obj.predict(transformed_feature)

        except Exception as e:
            raise CustomerException(e, sys) from e