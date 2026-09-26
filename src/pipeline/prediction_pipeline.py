import pandas as pd
import sys

from src.entity.config_entity import Prediction_config, ModelTrainerConfig, PredictionPipelineConfig
from src.utils.main_utils import MainUtils
from src.exception import CustomerException
from src.logger import logging
from src.ml.model.s3_estimator import CustomerClusterEstimator

class CustomerData:
    def get_input_dataset(
        self,
        column_schema: dict,
        input_data
    ):
        columns = column_schema.keys()

        input_dataset = pd.DataFrame(
            [input_data],
            columns= columns
        )

        for key, value in column_schema.items():
            input_dataset[key] = input_dataset[key].astype(value)

        return input_dataset

    @staticmethod
    def create_input_dataframe(data):
        prediction_config = Prediction_config()
        prediction_schema = prediction_config.__dict__
        column_schema = prediction_schema['prediction_schema']['columns']

        customer_data = CustomerData()
        input_df = customer_data.get_input_dataset(
            column_schema= column_schema,
            input_data= data
        )

        return input_df

class PredictionPipeline:
    def __init__(self):
        self.utils = MainUtils()

    def prepare_input_data(
        self,
        input_data: list
    )  -> pd.DataFrame:
        try:
            customer_df =  CustomerData.create_input_dataframe(
                data= input_data
            )

            logging.info("Customer dataframe created.")
            return customer_df
        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_trained_model(
        self,
        model_trainer_config =  ModelTrainerConfig
    ):
        try:
            prediction_config = PredictionPipelineConfig()
            model = CustomerClusterEstimator(
                bucket_name= prediction_config.data_bucket_name,
                model_path= prediction_config.model_file_name
            )

            return model
        except Exception as e:
            raise CustomerException(e, sys) from e

    def run_pipeline(
        self,
        input_data: list
    ):
        try:
            input_df = self.prepare_input_data(input_data= input_data)
            model= self.get_trained_model()
            prediction = model.predict(input_df)

            return prediction
        except Exception as e:
            raise CustomerException(e, sys) from e