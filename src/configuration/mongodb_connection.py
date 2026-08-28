import os
import sys
import certifi
import pymongo

from src.constant.database import DATABASE_NAME
from src.constant.env_variable import MONGODB_URL_KEY
from src.exception import CustomerException

ca = certifi.where()

class MongoDBClient:
    client = None
    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongodb_url = MONGODB_URL_KEY
                if mongodb_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongodb_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name= database_name
        except Exception as e:
            raise CustomerException(e, sys) from e