import sys
import pandas as pd
from pandas import DataFrame
from evidently import Report
from evidently.presets import DataDriftPreset
from typing import Tuple

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.utils.main_utils import MainUtils
from src.exception import CustomerException
from src.logger import logging


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self.utils = MainUtils()
        self._schema_config= self.utils.read_schema_config_file()


    def detect_dataset_drift(
        self,
        reference_df: DataFrame,
        current_df: DataFrame
    ) -> bool:
    
        try:
            report = Report(
                metrics=[
                    DataDriftPreset()
                ]
            )

            result = report.run(
                reference_data=reference_df,
                current_data=current_df
            )
            report_dict = result.dict()

            self.utils.write_yaml_file(
                file_path= self.data_validation_config.drift_report_file_path,
                content= report_dict
            )
            n_features = len(reference_df.columns)

            drift_metric = next(
                metric
                for metric in report_dict["metrics"]
                if metric["metric_name"].startswith("DriftedColumnsCount")
            )

            n_drifted_features = int(
                drift_metric["value"]["count"]
            )

            drift_share = drift_metric["value"]["share"]

            dataset_drift = drift_share >= 0.5

            logging.info(
                f"{n_drifted_features}/{n_features} features drifted "
                f"({drift_share:.2%})"
            )

            logging.info(f"Dataset drift detected: {dataset_drift}")

            return dataset_drift

        except Exception as e:
            raise CustomerException(e, sys) from e

    def validate_schema_columns(
        self,
        df: DataFrame
    ) -> bool:
        try:
            expected_columns = set(self._schema_config["columns"].keys())
            actual_columns = set(df.columns)
            status = expected_columns == actual_columns
            return status
        except Exception as e:
            raise CustomerException(e , sys) from e

    def validate_dataset_schema_columns(
        self,
        train_set: DataFrame,
        test_set: DataFrame
    ) -> Tuple[bool, bool]:
        try:
            train_schema_status = self.validate_schema_columns(train_set)
            test_schema_status = self.validate_schema_columns(test_set)

            logging.info("Validated schema columns on train and test set")

            return train_schema_status, test_schema_status
        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def read_data(file_path :str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e, sys) from e
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        logging.info("Starting DataValidation.initiate_data_validation")
        try:

            train_df, test_df =  (
                DataValidation.read_data(self.data_ingestion_artifact.trained_file_path),
                DataValidation.read_data(self.data_ingestion_artifact.test_file_path)
            )
            drift = self.detect_dataset_drift(train_df, test_df)

            (
                schema_train_col_status,
                schema_test_col_status
            ) = self.validate_dataset_schema_columns(train_set= train_df, test_set=test_df)
            

            logging.info(
                f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}"
            )

            if(
                schema_test_col_status is True and
                schema_test_col_status is True and
                drift is False
            ):
                validation_status = True
            else:
                validation_status = False
            
            data_validation_artifact = DataValidationArtifact(
                validation_status= validation_status,
                valid_train_file_path= self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path= self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path= self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path= self.data_validation_config.invalid_test_file_path,
                drift_report_file_path= self.data_validation_config.drift_report_file_path
            )

            logging.info("Exiting DataValidation.initiate_data_validation")
            return data_validation_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e