import sys
import yaml
import os
import pickle
import numpy as np

from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.exception import CustomerException

class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "r", encoding="utf-8") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomerException(e, sys) from e


    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(SCHEMA_FILE_PATH)

            return schema_config

        except Exception as e:
            raise CustomerException(e, sys) from e


    def write_yaml_file(
        self,
        file_path: str,
        content: object,
        replace: bool = False
    ) -> None:
        try:
            if replace and os.path.exists(file_path):
                os.remove(file_path)

            os.makedirs(
                os.path.dirname(file_path),
                exist_ok=True
            )

            with open(file_path, "w", encoding="utf-8") as file:
                yaml.dump(
                    content,
                    file,
                    sort_keys=False
                )

        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def save_object(file_path: str, object: object) -> None:
        try:
            with open(file_path, 'wb') as file_obj:
                pickle.dump(object, file_obj)

        except Exception as e:
            raise CustomerException(e, sys) from e

    def save_numpy_array_data(
        self,
        file_path: str,
        array: np.array
    ):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok= True)
            with open(file_path, 'wb') as file_obj:
                np.save(file_obj, array)
        except Exception as e:
            raise CustomerException(e, sys) from e

    def load_numpy_array_data(self, file_path: str) -> np.array:
        try:
            with open(file_path, 'rb') as file_obj:
                return np.load(file_obj)
        except Exception as e:
            raise CustomerException(e, sys) from e

    def load_object(self,file_path: str) -> object:
        try:
            with open(file_path,'rb') as file_obj:
                obj = pickle.load(file_obj)
                
            return obj
        except Exception as e:
            raise CustomerException(e, sys) from e