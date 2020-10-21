# import json
# import os

# import boto3
# import pandas as pd

# import pytest
# import woodwork.serialize as serialize
import woodwork.deserialize as deserialize
from woodwork import DataTable
from woodwork.logical_types import Datetime, Ordinal

# BUCKET_NAME = "test-bucket"
# WRITE_KEY_NAME = "test-key"
# TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
# TEST_FILE = "test_serialization_data_datatable_schema_1.0.0.tar"
# S3_URL = "s3://woodwork-static/" + TEST_FILE
# URL = "https://woodwork-static.s3.amazonaws.com/" + TEST_FILE
# TEST_KEY = "test_access_key_es"


def test_save_table_metadata(sample_df, tmpdir):
    date = Datetime(datetime_format='%Y-%m-%d')
    dt = DataTable(sample_df,
                   index='id',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'signup_date': date, 'age': Ordinal(order=[25, 33])})

    path = str(tmpdir)

    dt.save_metadata(path)
    serialized = dt.get_metadata()
    deserialized = deserialize.read_table_metadata(path)

    for key in serialized.keys():
        assert serialized[key] == deserialized[key]


# test saving locally
# test storing and retreiving from s3 the same dt
#  test retreiving a different dt from s3
