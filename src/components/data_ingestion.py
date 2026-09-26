import os
import sys
from pandas import DataFrame
from typing import Tuple

from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig
from src.utils.main_utils import MainUtils
from src.logger import logging
from src.exception import CustomerException
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.customer_data import CustomerData
from src.constant.database import COLLECTION_NAME


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig  =  DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config
        self.utils = MainUtils()

    def split_data_as_train_test(self, dataframe:DataFrame) -> Tuple[DataFrame, DataFrame]:
        logging.info("Starting DataIngestion.split_data_as_train_test")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logging.info("Data spiltting done.")
            ingested_data_dir = self.data_ingestion_config.ingested_data_dir
            os.makedirs(ingested_data_dir,exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logging.info("Train and test data saved. Exiting DataIngestion.split_data_as_train_test")
        except Exception as e:
            raise CustomerException(e, sys) from e

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Starting DataIngestion.export_data_into_feature_store")
            customer_data = CustomerData()
            customer_dataframe= customer_data.export_collection_as_dataframe(
                collection_name= COLLECTION_NAME
            )

            logging.info(f"Shape of dataframe: {customer_dataframe.shape}")
            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")

            customer_dataframe.to_csv(
                feature_store_file_path,
                index=False,header=True
            )

            return customer_dataframe
        
        except Exception as e:
            raise CustomerException(e, sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            logging.info("Starting DataIngestion.initiate_data_ingestion")
            dataframe = self.export_data_into_feature_store()
            _schema_config = self.utils.read_schema_config_file()
            dataframe = dataframe.drop(_schema_config["drop_columns"], axis=1)
            logging.info("Got the data from mongodb")
            self.split_data_as_train_test(dataframe)


            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path= self.data_ingestion_config.training_file_path,
                test_file_path= self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            logging.info("Exiting DataIngestion.initiate_data_ingestion")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e    