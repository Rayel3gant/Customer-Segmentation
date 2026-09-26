import sys
from mypy_boto3_s3.service_resource import Bucket
from typing import Union , List
from io import StringIO
import pickle
import os

from src.configuration.aws_connection import S3Client
from src.exception import CustomerException
from src.logger import logging

class S3BucketOperations:
    def __init__(self):
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(
        self,
        bucket_name: str,
        s3_key: str
    ):
        try:
            bucket= self.get_bucket(bucket_name= bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            if(len(file_objects) > 0):
                return True
            else:
                return False
        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_bucket(self, bucket_name:str) -> Bucket:
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            return bucket
        except Exception as e:
            raise CustomerException(e, sys) from e 

    def get_file_object(
        self,
        file_name: str,
        bucket_name: str
    )-> Union[List[object], object]:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=file_name)]
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(file_objects)
            return file_objs
        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def read_object(
        object_name: str,
        decode: bool = True,
        make_readable: bool = False
    ) -> Union[StringIO, str]:
        try:
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode is True
                else object_name.get()["Body"].read()
            )
            conv_func = lambda: StringIO(func()) if make_readable is True else func()
            return conv_func()
        except Exception as e:
            raise CustomerException(e, sys) from e

    def load_model(
        self,
        model_name: str,
        bucket_name: str,
        model_dir: str = None
    ) -> object:
        try:
            fun = (
                lambda: model_name
                if model_dir is None
                else model_dir + "/"+ model_name
            )
            model_file = fun()
            file_object = self.get_file_object(
                file_name= model_file,
                bucket_name= bucket_name
            )
            model_object = self.read_object(
                object_name= file_object,
                decode= False
            )
            model = pickle.loads(model_object)
            return model
        except Exception as e:
            raise CustomerException(e, sys) from e

    def upload_file(
        self,
        from_filename: str,
        to_filename: str,
        bucket_name: str,
        remove: bool = True
    ): 
        try:
            logging.info(
                f"Uploading {from_filename} file to {to_filename} file in {bucket_name} bucket"
            )

            self.s3_resource.meta.client.upload_file(
                from_filename,
                bucket_name,
                to_filename
            )

            if remove is True:
                os.remove(from_filename)

            logging.info("File Uploaded to bucket.")

        except Exception as e:
            raise CustomerException(e, sys) from e