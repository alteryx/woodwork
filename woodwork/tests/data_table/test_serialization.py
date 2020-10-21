# import json
# import os

# import boto3
# import pandas as pd

# import pytest
# import woodwork.serialize as serialize
# import woodwork.deserialize as deserialize
from woodwork import DataTable
from woodwork.logical_types import Ordinal

# BUCKET_NAME = "test-bucket"
# WRITE_KEY_NAME = "test-key"
# TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
# TEST_FILE = "test_serialization_data_datatable_schema_1.0.0.tar"
# S3_URL = "s3://woodwork-static/" + TEST_FILE
# URL = "https://woodwork-static.s3.amazonaws.com/" + TEST_FILE
# TEST_KEY = "test_access_key_es"


def test_to_dictionary(sample_df):
    expected = {'woodwork_version': '0.0.3',
                'schema_version': '1.0.0',
                'name': 'test_data',
                'index': 'id',
                'time_index': None,
                'metadata': [{'name': 'id',
                              'ordinal': 0,
                              'logical_type': {'parameters': {}, 'type': 'WholeNumber'},
                              'physical_type': {'type': 'Int64'},
                              'semantic_tags': ['index', 'tag1']},
                             {'name': 'full_name',
                              'ordinal': 1,
                              'logical_type': {'parameters': {}, 'type': 'NaturalLanguage'},
                              'physical_type': {'type': 'string'},
                              'semantic_tags': []},
                             {'name': 'email',
                              'ordinal': 2,
                              'logical_type': {'parameters': {}, 'type': 'NaturalLanguage'},
                              'physical_type': {'type': 'string'},
                              'semantic_tags': []},
                             {'name': 'phone_number',
                              'ordinal': 3,
                              'logical_type': {'parameters': {}, 'type': 'NaturalLanguage'},
                              'physical_type': {'type': 'string'},
                              'semantic_tags': []},
                             {'name': 'age',
                              'ordinal': 4,
                              'logical_type': {'parameters': {'order': [25, 33]}, 'type': 'Ordinal'},
                              'physical_type': {'type': 'category'},
                              'semantic_tags': ['category']},
                             {'name': 'signup_date',
                              'ordinal': 5,
                              'logical_type': {'parameters': {'datetime_format': None},
                                               'type': 'Datetime'},
                              'physical_type': {'type': 'datetime64[ns]'},
                              'semantic_tags': []},
                             {'name': 'is_registered',
                              'ordinal': 6,
                              'logical_type': {'parameters': {}, 'type': 'Boolean'},
                              'physical_type': {'type': 'boolean'},
                              'semantic_tags': []}]}
    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33])})
    metadata = dt.to_dictionary()

    assert metadata.__eq__(expected)


# test saving locally
# test storing and retreiving from s3 the same dt
#  test retreiving a different dt from s3
