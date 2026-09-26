import sys
import pandas as pd
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score, recall_score , precision_score
from typing import Optional

from src.entity.config_entity import ModelEvaluationConfig, Prediction_config
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ClassificationMetricArtifact
from src.utils.main_utils import MainUtils
from src.exception import CustomerException
from src.logger import logging
from src.ml.model.s3_estimator import CustomerClusterEstimator

def calculation_metric(
    model,
    X,
    y
) -> ClassificationMetricArtifact:
    y_pred = model.predict(X)

    classification_metric = ClassificationMetricArtifact(
        f1_score= f1_score(
            y_pred= y_pred,
            y_true= y,
            average= 'weighted'
        ),
        recall_score= recall_score(
            y_pred=y_pred,
            y_true= y,
            average='weighted'
        ),
        precision_score= precision_score(
            y_pred=y_pred,
            y_true= y,
            average='weighted'
        )
    )
    return classification_metric

@dataclass
class EvaluationModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    changed_accuracy: float
    best_model_metric_artifact: ClassificationMetricArtifact

def convert_test_numpy_array_to_dataframe(array: np.ndarray):
    prediction_config = Prediction_config().__dict__
    columns = prediction_config['prediction_schema']['columns'].keys()  
    dataframe = pd.DataFrame(array, columns=columns)
    return dataframe

class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        self.model_evaluation_config= model_evaluation_config
        self.data_ingestion_artifact= data_ingestion_artifact
        self.data_transformation_artifact= data_transformation_artifact
        self.model_trainer_artifact= model_trainer_artifact
        self.utils= MainUtils()

    def get_best_model(self) -> Optional[CustomerClusterEstimator]:
        try:
            bucket_name = self.model_evaluation_config.bucket_name
            model_path = self.model_evaluation_config.s3_model_key_path

            customer_cluster_estimator = CustomerClusterEstimator(
                bucket_name= bucket_name,
                model_path= model_path
            )
            if customer_cluster_estimator.is_model_present(model_path= model_path) :
                return customer_cluster_estimator
            return None
        except Exception as e:
            raise CustomerException(e, sys) from e

    def evaluate_model(self) -> EvaluationModelResponse:
        try:
            test_arr= self.utils.load_numpy_array_data(
                file_path= self.data_transformation_artifact.transformed_test_file_path
            )

            X_test = convert_test_numpy_array_to_dataframe(array=pd.DataFrame(test_arr[:,:-1]))
            y_test = pd.DataFrame(test_arr[:,-1])

            trained_model = self.utils.load_object(
                file_path= self.model_trainer_artifact.trained_model_file_path
            )

            y_pred = trained_model.predict(X_test)
            trained_model_f1_score = f1_score(
                y_pred= y_pred,
                y_true= y_test,
                average='weighted'
            )

            best_model_f1_score = None
            best_model_metric_artifact = None
            best_model = self.get_best_model()

            if best_model is not None:
                y_pred_best_model = best_model.predict(X_test)
                best_model_f1_score = f1_score(
                    y_pred= y_pred_best_model,
                    y_true=y_test,
                    average= 'weighted'
                )
                best_model_metric_artifact= calculation_metric(
                    model= best_model,
                    X= X_test,
                    y= y_test
                )

            tmp_best_model_score= 0 if best_model_f1_score is None else best_model_f1_score
            model_response = EvaluationModelResponse(
                trained_model_f1_score= trained_model_f1_score,
                best_model_f1_score= best_model_f1_score,
                is_model_accepted= trained_model_f1_score > tmp_best_model_score,
                changed_accuracy= trained_model_f1_score - tmp_best_model_score,
                best_model_metric_artifact= best_model_metric_artifact
            )

            return model_response

        except Exception as e:
            raise CustomerException(e, sys) from e
        
    def initiate_model_evaluation(self):
        try:
            logging.info("Starting ModelEvaluation.initiate_model_evaluation")
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted= evaluate_model_response.is_model_accepted,
                best_model_path = self.model_trainer_artifact.trained_model_file_path,
                trained_model_path = self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy = evaluate_model_response.changed_accuracy,
                best_model_metric_artifact = evaluate_model_response.best_model_metric_artifact
            )
       
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e