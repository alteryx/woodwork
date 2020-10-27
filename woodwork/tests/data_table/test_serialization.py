import json
import os

import boto3
import pandas as pd
import pytest

import woodwork.deserialize as deserialize
import woodwork.serialize as serialize
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


def xfail_not_pandas(dataframe):
    # TODO: Allow serialization of non-pandas dataframes
    if not isinstance(dataframe, pd.DataFrame):
        pytest.xfail('fails with Dask - Dask serialization not yet implemented')


def test_to_dictionary(sample_df):
    xfail_not_pandas(sample_df)
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
                              'logical_type': {'parameters': {'order': [25, 33, 57]}, 'type': 'Ordinal'},
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
                   logical_types={'age': Ordinal(order=[25, 33, 57])})
    metadata = dt.to_dictionary()

    assert metadata.__eq__(expected)


def test_serialize_wrong_format(sample_df, tmpdir):
    xfail_not_pandas(sample_df)
    dt = DataTable(sample_df)

    error = 'must be one of the following formats: csv'
    with pytest.raises(ValueError, match=error):
        serialize.write_datatable(dt, str(tmpdir), format='test')


def test_to_csv(sample_df, tmpdir):
    xfail_not_pandas(sample_df)
    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33, 57])})
    dt.to_csv(str(tmpdir), encoding='utf-8', engine='python')

    _dt = deserialize.read_datatable(str(tmpdir))

    pd.testing.assert_frame_equal(dt.to_dataframe(), _dt.to_dataframe())
    assert dt.name == _dt.name
    assert dt.index == _dt.index
    assert dt.time_index == _dt.time_index
    assert dt.columns.keys() == _dt.columns.keys()

    for col_name in dt.columns.keys():
        col = dt[col_name]
        _col = _dt[col_name]

        assert _get_ltype_class(col.logical_type) == _get_ltype_class(_col.logical_type)
        assert _get_ltype_params(col.logical_type) == _get_ltype_params(_col.logical_type)
        assert col.semantic_tags == _col.semantic_tags
        assert col.name == _col.name
        assert col.dtype == _col.dtype


@pytest.fixture
def s3_client():
    # TODO: Fix Moto tests needing to explicitly set permissions for objects
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
    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')


def test_to_csv_S3(sample_df, s3_client, s3_bucket):
    xfail_not_pandas(sample_df)
    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33, 57])})
    dt.to_csv(TEST_S3_URL, encoding='utf-8', engine='python')
    make_public(s3_client, s3_bucket)

    _dt = deserialize.read_datatable(TEST_S3_URL)

    pd.testing.assert_frame_equal(dt.to_dataframe(), _dt.to_dataframe())
    assert dt.name == _dt.name
    assert dt.index == _dt.index
    assert dt.time_index == _dt.time_index
    assert dt.columns.keys() == _dt.columns.keys()

    for col_name in dt.columns.keys():
        col = dt[col_name]
        _col = _dt[col_name]

        assert _get_ltype_class(col.logical_type) == _get_ltype_class(_col.logical_type)
        assert _get_ltype_params(col.logical_type) == _get_ltype_params(_col.logical_type)
        assert col.semantic_tags == _col.semantic_tags
        assert col.name == _col.name
        assert col.dtype == _col.dtype


def test_to_csv_S3_anon(sample_df, s3_client, s3_bucket):
    xfail_not_pandas(sample_df)
    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   time_index='signup_date',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33, 57])})
    dt.to_csv(TEST_S3_URL, encoding='utf-8', engine='python', profile_name=False)
    make_public(s3_client, s3_bucket)

    _dt = deserialize.read_datatable(TEST_S3_URL, profile_name=False)

    pd.testing.assert_frame_equal(dt.to_dataframe(), _dt.to_dataframe())
    assert dt.name == _dt.name
    assert dt.index == _dt.index
    assert dt.time_index == _dt.time_index
    assert dt.columns.keys() == _dt.columns.keys()

    for col_name in dt.columns.keys():
        col = dt[col_name]
        _col = _dt[col_name]

        assert _get_ltype_class(col.logical_type) == _get_ltype_class(_col.logical_type)
        assert _get_ltype_params(col.logical_type) == _get_ltype_params(_col.logical_type)
        assert col.semantic_tags == _col.semantic_tags
        assert col.name == _col.name
        assert col.dtype == _col.dtype


def create_test_credentials(test_path):
    with open(test_path, "w+") as f:
        f.write("[test]\n")
        f.write("aws_access_key_id=AKIAIOSFODNN7EXAMPLE\n")
        f.write("aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n")


def create_test_config(test_path_config):
    with open(test_path_config, "w+") as f:
        f.write("[profile test]\n")
        f.write("region=us-east-2\n")
        f.write("output=text\n")


@pytest.fixture
def setup_test_profile(monkeypatch, tmpdir):
    cache = str(tmpdir.join('.cache').mkdir())
    test_path = os.path.join(cache, 'test_credentials')
    test_path_config = os.path.join(cache, 'test_config')
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", test_path)
    monkeypatch.setenv("AWS_CONFIG_FILE", test_path_config)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "test")

    try:
        os.remove(test_path)
        os.remove(test_path_config)
    except OSError:
        pass

    create_test_credentials(test_path)
    create_test_config(test_path_config)
    yield
    os.remove(test_path)
    os.remove(test_path_config)


def test_s3_test_profile(sample_df, s3_client, s3_bucket, setup_test_profile):
    xfail_not_pandas(sample_df)
    dt = DataTable(sample_df)
    dt.to_csv(TEST_S3_URL, encoding='utf-8', engine='python', profile_name='test')
    make_public(s3_client, s3_bucket)
    _dt = deserialize.read_datatable(TEST_S3_URL, profile_name='test')

    pd.testing.assert_frame_equal(dt.to_dataframe(), _dt.to_dataframe())
    assert dt.name == _dt.name
    assert dt.index == _dt.index
    assert dt.time_index == _dt.time_index
    assert dt.columns.keys() == _dt.columns.keys()

    for col_name in dt.columns.keys():
        col = dt[col_name]
        _col = _dt[col_name]

        assert _get_ltype_class(col.logical_type) == _get_ltype_class(_col.logical_type)
        assert _get_ltype_params(col.logical_type) == _get_ltype_params(_col.logical_type)
        assert col.semantic_tags == _col.semantic_tags
        assert col.name == _col.name
        assert col.dtype == _col.dtype


def test_serialize_url_csv(sample_df):
    xfail_not_pandas(sample_df)
    dt = DataTable(sample_df)
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        dt.to_csv(URL, encoding='utf-8', engine='python')


def test_serialize_subdirs_not_removed(sample_df, tmpdir):
    xfail_not_pandas(sample_df)
    dt = DataTable(sample_df)
    write_path = tmpdir.mkdir("test")
    test_dir = write_path.mkdir("test_dir")
    with open(str(write_path.join('table_metadata.json')), 'w') as f:
        json.dump('__SAMPLE_TEXT__', f)
    compression = None
    serialize.write_datatable(dt, path=str(write_path), index='1', sep='\t', encoding='utf-8', compression=compression)
    assert os.path.exists(str(test_dir))
    with open(str(write_path.join('table_metadata.json')), 'r') as f:
        assert '__SAMPLE_TEXT__' not in json.load(f)


def test_deserialize_url_csv(sample_df):
    # should fail til lwe update
    xfail_not_pandas(sample_df)
    dt = DataTable(sample_df, index='id')
    _dt = deserialize.read_datatable(URL)

    pd.testing.assert_frame_equal(dt.to_dataframe(), _dt.to_dataframe())
    assert dt.name == _dt.name
    assert dt.index == _dt.index
    assert dt.time_index == _dt.time_index
    assert dt.columns.keys() == _dt.columns.keys()

    for col_name in dt.columns.keys():
        col = dt[col_name]
        _col = _dt[col_name]

        assert _get_ltype_class(col.logical_type) == _get_ltype_class(_col.logical_type)
        assert _get_ltype_params(col.logical_type) == _get_ltype_params(_col.logical_type)
        assert col.semantic_tags == _col.semantic_tags
        assert col.name == _col.name
        assert col.dtype == _col.dtype


def test_deserialize_url_csv_anon(sample_df):
    # should fail til lwe update
    xfail_not_pandas(sample_df)
    dt = DataTable(sample_df, index='id')
    _dt = deserialize.read_datatable(URL, profile_name=False)

    pd.testing.assert_frame_equal(dt.to_dataframe(), _dt.to_dataframe())
    assert dt.name == _dt.name
    assert dt.index == _dt.index
    assert dt.time_index == _dt.time_index
    assert dt.columns.keys() == _dt.columns.keys()

    for col_name in dt.columns.keys():
        col = dt[col_name]
        _col = _dt[col_name]

        assert _get_ltype_class(col.logical_type) == _get_ltype_class(_col.logical_type)
        assert _get_ltype_params(col.logical_type) == _get_ltype_params(_col.logical_type)
        assert col.semantic_tags == _col.semantic_tags
        assert col.name == _col.name
        assert col.dtype == _col.dtype


def test_check_later_schema_version():
    def test_version(major, minor, patch, raises=True):
        version_to_check = '.'.join([str(v) for v in [major, minor, patch]])
        if raises:
            warning_text = ('The schema version of the saved woodwork.DataTable '
                            '%s is greater than the latest supported %s. '
                            'You may need to upgrade woodwork. Attempting to load woodwork.DataTable ...'
                            % (version_to_check, serialize.SCHEMA_VERSION))
            with pytest.warns(UserWarning, match=warning_text):
                deserialize._check_schema_version(version_to_check)
        else:
            with pytest.warns(None) as record:
                deserialize._check_schema_version(version_to_check)
            assert len(record) == 0

    major, minor, patch = [int(s) for s in serialize.SCHEMA_VERSION.split('.')]

    test_version(major + 1, minor, patch)
    test_version(major, minor + 1, patch)
    test_version(major, minor, patch + 1)
    test_version(major, minor - 1, patch + 1, raises=False)


def test_earlier_schema_version():
    def test_version(major, minor, patch, raises=True):
        version_to_check = '.'.join([str(v) for v in [major, minor, patch]])
        if raises:
            warning_text = ('The schema version of the saved woodwork.DataTable '
                            '%s is no longer supported by this version '
                            'of woodwork. Attempting to load woodwork.DataTable ...'
                            % (version_to_check))
            with pytest.warns(UserWarning, match=warning_text):
                deserialize._check_schema_version(version_to_check)
        else:
            with pytest.warns(None) as record:
                deserialize._check_schema_version(version_to_check)
            assert len(record) == 0

    major, minor, patch = [int(s) for s in serialize.SCHEMA_VERSION.split('.')]

    test_version(major - 1, minor, patch)
    test_version(major, minor - 1, patch, raises=False)
    test_version(major, minor, patch - 1, raises=False)
