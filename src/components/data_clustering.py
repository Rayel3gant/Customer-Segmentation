import pandas as pd
import sys
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.entity.config_entity import PCAConfig
from src.exception import CustomerException
from src.logger import logging
from src.constant.training_pipeline import TARGET_COLUMN

class CreateClusters:
    def __init__(self):
        self.pca_config = PCAConfig()

    def optimize_dataset_using_pca(
        self,
        preprocessed_data= pd.DataFrame
    ) -> pd.DataFrame:
        try:
            pca_obj = PCA(**self.pca_config.__dict__).fit(preprocessed_data)
            reduced_dataset = pca_obj.fit_transform(preprocessed_data)

            logging.info("PCA Finised")
            return reduced_dataset
        except Exception as e:
            raise CustomerException(e, sys) from e

    def initialize_clustering(
        self,
        preprocessed_data: pd.DataFrame
    ) -> pd.DataFrame :
        try:
            logging.info("Starting CreateClusters.initialize_clustering")
            reduced_dataset= self.optimize_dataset_using_pca(preprocessed_data)

            model = KMeans(n_clusters= 3).fit(reduced_dataset)
            preprocessed_data[TARGET_COLUMN] = model.labels_.astype(int)
            logging.info("Clustering is Finished")
            return preprocessed_data
        except Exception as e:
            raise CustomerException(e, sys) from e