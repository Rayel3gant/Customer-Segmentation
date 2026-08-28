import sys
import yaml

from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.exception import CustomerException

class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomerException(e, sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(SCHEMA_FILE_PATH)

            return schema_config

        except Exception as e:
            raise CustomerException(e, sys) from e