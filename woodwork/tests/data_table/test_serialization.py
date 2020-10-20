import json
import os

import boto3
import pandas as pd
import pytest

BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_serialization_data_datatable_schema_1.0.0.tar"
S3_URL = "s3://woodwork-static/" + TEST_FILE
URL = "https://woodwork-static.s3.amazonaws.com/" + TEST_FILE
TEST_KEY = "test_access_key_es"
