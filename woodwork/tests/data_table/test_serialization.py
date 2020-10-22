import os

import boto3
import pandas as pd
import pytest

import woodwork.deserialize as deserialize
from woodwork import DataTable
from woodwork.logical_types import Ordinal
from woodwork.utils import _get_ltype_class, _get_ltype_params

BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_serialization_data_datatable_schema_1.0.0.tar"
S3_URL = "s3://woodwork-static/" + TEST_FILE
URL = "https://woodwork-static.s3.amazonaws.com/" + TEST_FILE
TEST_KEY = "test_access_key_es"


def test_to_dictionary(sample_df):
    expected = {'schema_version': '1.0.0',
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


def test_to_csv(sample_df, tmpdir):
    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33])})
    dt.to_csv(str(tmpdir), encoding='utf-8', engine='python')

    _dt = deserialize.read_datatable(str(tmpdir))

    pd.testing.assert_frame_equal(dt.to_pandas(), _dt.to_pandas())
    assert dt.name == _dt.name
    assert dt.index == _dt.index
    assert dt.time_index == _dt.time_index
    assert dt.columns.keys() == _dt.columns.keys()

    for col_name in dt.columns.keys():
        col = dt[col_name]
        _col = _dt[col_name]

        # --> this might be a better way to do ltype equality
        assert _get_ltype_class(col.logical_type) == _get_ltype_class(_col.logical_type)
        assert _get_ltype_params(col.logical_type) == _get_ltype_params(_col.logical_type)
        assert col.semantic_tags == _col.semantic_tags
        assert col.name == _col.name
        assert col.dtype == _col.dtype

# TODO: Fix Moto tests needing to explicitly set permissions for objects


@pytest.fixture
def s3_client():
    _environ = os.environ.copy()
    from moto import mock_s3
    with mock_s3():
        s3 = boto3.resource('s3')
        yield s3
    os.environ.clear()
    os.environ.update(_environ)


@pytest.fixture
def s3_bucket(s3_client):
    s3_client.create_bucket(Bucket=BUCKET_NAME, ACL='public-read-write')
    s3_bucket = s3_client.Bucket(BUCKET_NAME)
    yield s3_bucket


def make_public(s3_client, s3_bucket):
    obj = list(s3_bucket.objects.all())[0].keyx
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')


def test_to_csv_S3(sample_df, s3_client, s3_bucket):
    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33])})
    dt.to_csv(TEST_S3_URL, encoding='utf-8', engine='python')
    make_public(s3_client, s3_bucket)

    _dt = deserialize.read_datatable(TEST_S3_URL)

    pd.testing.assert_frame_equal(dt.to_pandas(), _dt.to_pandas())
    assert dt.name == _dt.name
    assert dt.index == _dt.index
    assert dt.time_index == _dt.time_index
    assert dt.columns.keys() == _dt.columns.keys()

    for col_name in dt.columns.keys():
        col = dt[col_name]
        _col = _dt[col_name]

        # --> this might be a better way to do ltype equality
        assert _get_ltype_class(col.logical_type) == _get_ltype_class(_col.logical_type)
        assert _get_ltype_params(col.logical_type) == _get_ltype_params(_col.logical_type)
        assert col.semantic_tags == _col.semantic_tags
        assert col.name == _col.name
        assert col.dtype == _col.dtype


def test_serialize_url_csv(sample_df):
    dt = DataTable(sample_df)
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        dt.to_csv(URL, encoding='utf-8', engine='python')
