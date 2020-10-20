# import json
# import os

# import boto3
import pandas as pd

# import pytest
# import woodwork.serialize as serialize
import woodwork.deserialize as deserialize
from woodwork import DataTable
from woodwork.logical_types import Datetime

# BUCKET_NAME = "test-bucket"
# WRITE_KEY_NAME = "test-key"
# TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
# TEST_FILE = "test_serialization_data_datatable_schema_1.0.0.tar"
# S3_URL = "s3://woodwork-static/" + TEST_FILE
# URL = "https://woodwork-static.s3.amazonaws.com/" + TEST_FILE
# TEST_KEY = "test_access_key_es"


def test_table_metadata():
    df = pd.DataFrame({
        'ints': pd.Series([1, 2, 3]),
        'bools': [True, False, False],
        'categories': pd.Series(['hi', 'bye', None]),
        'dates': pd.Series(['2020-01-01', '2020-01-02', '2020-01-03']),
        'dates2': pd.Series(['2020-01-01', '2020-01-02', '2020-01-03']),
    })
    date = Datetime(datetime_format='%Y-%m-%d')
    dt = DataTable(df, index='ints', semantic_tags={'ints': 'tag1'}, logical_types={'dates': date})

    path = '../datatables/testing-dir'  # --> need to use a generalized tmpdir

    dt.save_metadata(path)
    serialized = dt.get_metadata()
    deserialized = deserialize.read_table_metadata(path)

    for key in serialized.keys():
        assert serialized[key] == deserialized[key]

# test saving locally
# test storing and retreiving from s3 the same dt
#  test retreiving a different dt from s3
