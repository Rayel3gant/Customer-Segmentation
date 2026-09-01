import pandas as pd
import sys, os
from datetime import datetime
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact , DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig, SimpleImputerConfig
from src.components.data_ingestion import DataIngestion
from src.utils.main_utils import MainUtils
from src.exception import CustomerException
from src.logger import logging
from src.components.data_clustering import CreateClusters
from src.constant.training_pipeline import TARGET_COLUMN

class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self.data_ingestion = DataIngestion()
        self.imputer_config = SimpleImputerConfig()
        self.utils = MainUtils()


    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_new_features(
        self,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame
    )-> pd.DataFrame:

        train_set_with_new_features = pd.DataFrame()
        test_set_with_new_features = pd.DataFrame()

        datasets = {
            "train_set": train_set,
            "test_set": test_set
        }

        for key in datasets:
            dataset = datasets[key]

            dataset['Age'] = 2026 - dataset['Year_Birth']

            dataset['Education'] = dataset['Education'].replace({
                "Basic": 0,
                "2n Cycle": 1,
                "Graduation": 2,
                "Master": 3,
                "PhD": 4
            })

            dataset['Marital_Status'] = dataset['Marital_Status'].replace({
                "Married": 1,
                "Together": 1,
                "Absurd": 0,
                "Widow": 0,
                "YOLO": 0,
                "Divorced": 0,
                "Single": 0,
                "Alone": 0
            })

            dataset['Children'] = dataset['Kidhome'] + dataset['Teenhome']

            dataset['Family_Size'] = dataset['Marital_Status'] + dataset['Children'] + 1

            dataset['Total_Spending'] = dataset['MntWines'] + dataset['MntFruits'] + dataset['MntMeatProducts'] + dataset['MntFishProducts'] + dataset['MntSweetProducts'] + dataset['MntGoldProds']

            dataset['Total_Promo'] = dataset['AcceptedCmp1'] + dataset['AcceptedCmp2'] + dataset['AcceptedCmp3'] + dataset['AcceptedCmp4'] + dataset['AcceptedCmp5']

            dataset['Dt_Customer'] = pd.to_datetime(dataset['Dt_Customer'],format='%d-%m-%Y')
            today = datetime.today()
            dataset['Days_as_Customer'] = (today - dataset['Dt_Customer']).dt.days

            dataset['Offers_Responded_To'] = dataset['Total_Promo'] + dataset['Response']

            dataset['Parental_Status'] = np.where(dataset['Children']>0 , 1 , 0)

            columns_to_drop = ['Year_Birth','Kidhome','Teenhome']
            dataset = dataset.drop(columns=columns_to_drop)
            dataset = dataset.rename(columns={
                "MntWines": "Wines",
                "MntFruits":"Fruits",
                "MntMeatProducts":"Meat",
                "MntFishProducts":"Fish",
                "MntSweetProducts":"Sweets",
                "MntGoldProds":"Gold",
                "NumWebPurchases": "Web",
                "NumCatalogPurchases":"Catalog",
                "NumStorePurchases":"Store",
                "NumDealsPurchases":"Discount_Purchases"
            })

            dataset = dataset[
                ["Age","Education","Marital_Status","Parental_Status",
                "Children","Income","Total_Spending","Days_as_Customer",
                "Recency","Wines","Fruits","Meat","Fish","Sweets","Gold",
                "Web","Catalog","Store","Discount_Purchases","Total_Promo",
                "NumWebVisitsMonth"]]

            if key == 'train_set':
                train_set_with_new_features = pd.concat([train_set_with_new_features, dataset])
            else:
                test_set_with_new_features = pd.concat([test_set_with_new_features, dataset])

        logging.info("New Features created")
        return train_set_with_new_features, test_set_with_new_features
        
    def transform_data(
        self,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame
    ) -> pd.DataFrame :
        logging.info("Starting DataTransformation.transform_data")
        try:

            numeric_features = [feature for feature in train_set.columns if train_set[feature].dtype != 'O']
            outlier_features = ["Wines","Fruits","Meat","Fish","Sweets","Gold","Age","Total_Spending"]
            numeric_features = [x for x in numeric_features if x not in outlier_features]

            logging.info("Initialized StandardScaler, SimpleImputer")

            numeric_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(**self.imputer_config.__dict__)),
                    ("StandardScaler",StandardScaler())
                ]
            )

            outlier_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(**self.imputer_config.__dict__)),
                    ("Transformer", PowerTransformer(standardize= True))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numeric pipeline",numeric_pipeline, numeric_features),
                    ("outlier pipeline",outlier_pipeline,outlier_features)
                ]
            )
        
            preprocessed_train_set = preprocessor.fit_transform(train_set)
            preprocessed_test_set = preprocessor.transform(test_set)
            feature_names = preprocessor.get_feature_names_out()

            preprocessed_train_set= pd.DataFrame(
                preprocessed_train_set,
                columns= feature_names
            )
            preprocessed_test_set = pd.DataFrame(
                preprocessed_test_set,
                columns= feature_names
            )

            preprocessor_obj_dir = os.path.dirname(
                self.data_transformation_config.transformed_object_file_path
            )
            os.makedirs(preprocessor_obj_dir, exist_ok= True)
            self.utils.save_object(
                file_path= self.data_transformation_config.transformed_object_file_path,
                object= preprocessor
            )
            logging.info("Saved Preprocessor object to {}".format(preprocessor_obj_dir))

            logging.info("Exiting DataTransformation.transform_data")
            return preprocessed_train_set, preprocessed_test_set
        except Exception as e:
            raise CustomerException(e, sys) from e

    def initiate_data_transformation(self) :
        try:
            logging.info("Starting DataTransformation.initiate_data_transformation")

            if self.data_validation_artifact.validation_status:

                train_set = DataTransformation.read_data(
                    file_path= self.data_ingestion_artifact.trained_file_path
                )
                test_set = DataTransformation.read_data(
                    file_path= self.data_ingestion_artifact.test_file_path
                )

                train_set, test_set = self.get_new_features(
                    train_set= train_set,
                    test_set= test_set
                )

                preprocessed_train_set, preprocessed_test_set = self.transform_data(
                    train_set= train_set,
                    test_set= test_set
                )
                logging.info("Recieved Preprocessor object")

                cluster_creator = CreateClusters()
                labelled_train_set= cluster_creator.initialize_clustering(
                    preprocessed_data= preprocessed_train_set
                )
                labelled_test_set = cluster_creator.initialize_clustering(
                    preprocessed_data= preprocessed_test_set
                )

                X_train = labelled_train_set.drop(columns=[TARGET_COLUMN])
                y_train = labelled_train_set[TARGET_COLUMN]
                X_test = labelled_test_set.drop(columns=[TARGET_COLUMN])
                y_test = labelled_test_set[TARGET_COLUMN]

                train_arr = np.c_[
                    np.array(X_train),np.array(y_train)
                ]
                test_arr = np.c_[
                    np.array(X_test),np.array(y_test)
                ]
                self.utils.save_numpy_array_data(
                    file_path= self.data_transformation_config.transformed_train_file_path,
                    array= train_arr
                )
                self.utils.save_numpy_array_data(
                    file_path= self.data_transformation_config.transformed_test_file_path,
                    array= test_arr
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path= self.data_transformation_config.transformed_test_file_path,
                    transformed_object_file_path= self.data_transformation_config.transformed_object_file_path
                )
                logging.info("Exiting DataTransformation.initiate_data_transformation")
                return data_transformation_artifact
            else:
                raise Exception("Data Validation Failed.")
        except Exception as e:
            raise CustomerException(e, sys) from e