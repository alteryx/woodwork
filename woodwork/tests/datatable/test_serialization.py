import json
import os

import boto3
import pandas as pd
import pytest

import woodwork.deserialize as deserialize
import woodwork.serialize as serialize
from woodwork import DataTable
from woodwork.exceptions import OutdatedSchemaWarning, UpgradeSchemaWarning
from woodwork.logical_types import Ordinal
from woodwork.tests.testing_utils import to_pandas
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')

BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_serialization_data_datatable_schema_5.0.0.tar"
S3_URL = "s3://woodwork-static/" + TEST_FILE
URL = "https://woodwork-static.s3.amazonaws.com/" + TEST_FILE
TEST_KEY = "test_access_key_es"


def xfail_tmp_disappears(dataframe):
    # TODO: tmp file disappears after deserialize step, cannot check equality with Dask
    if not isinstance(dataframe, pd.DataFrame):
        pytest.xfail('tmp file disappears after deserialize step, cannot check equality with Dask')


def test_to_dictionary(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        table_type = 'dask'
    elif ks and isinstance(sample_df, ks.DataFrame):
        table_type = 'koalas'
    else:
        table_type = 'pandas'

    if ks and isinstance(sample_df, ks.DataFrame):
        int_val = 'int64'
        cat_val = 'object'
        string_val = 'object'
        bool_val = 'bool'
    else:
        int_val = 'Int64'
        cat_val = 'category'
        string_val = 'string'
        bool_val = 'boolean'
    expected = {'schema_version': '5.0.0',
                'name': 'test_data',
                'index': 'id',
                'time_index': None,
                'column_metadata': [{'name': 'id',
                                     'ordinal': 0,
                                     'logical_type': {'parameters': {}, 'type': 'Integer'},
                                     'physical_type': {'type': int_val},
                                     'semantic_tags': ['index', 'tag1'],
                                     'description': None,
                                     'metadata':{'is_sorted': True}},
                                    {'name': 'full_name',
                                     'ordinal': 1,
                                     'logical_type': {'parameters': {}, 'type': 'NaturalLanguage'},
                                     'physical_type': {'type': string_val},
                                     'semantic_tags': [],
                                     'description': None,
                                     'metadata':{}},
                                    {'name': 'email',
                                     'ordinal': 2,
                                     'logical_type': {'parameters': {}, 'type': 'NaturalLanguage'},
                                     'physical_type': {'type': string_val},
                                     'semantic_tags': [],
                                     'description': None,
                                     'metadata':{}},
                                    {'name': 'phone_number',
                                     'ordinal': 3,
                                     'logical_type': {'parameters': {}, 'type': 'NaturalLanguage'},
                                     'physical_type': {'type': string_val},
                                     'semantic_tags': [],
                                     'description': None,
                                     'metadata': {}},
                                    {'name': 'age',
                                     'ordinal': 4,
                                     'logical_type': {'parameters': {'order': [25, 33, 57]}, 'type': 'Ordinal'},
                                     'physical_type': {'type': cat_val},
                                     'semantic_tags': ['category'],
                                     'description': 'age of the user',
                                     'metadata':{'interesting_values': [33, 57]}},
                                    {'name': 'signup_date',
                                     'ordinal': 5,
                                     'logical_type': {'parameters': {},
                                                      'type': 'Datetime'},
                                     'physical_type': {'type': 'datetime64[ns]'},
                                     'semantic_tags': [],
                                     'description': 'original signup date',
                                     'metadata':{}},
                                    {'name': 'is_registered',
                                     'ordinal': 6,
                                     'logical_type': {'parameters': {}, 'type': 'Boolean'},
                                     'physical_type': {'type': bool_val},
                                     'semantic_tags': [],
                                     'description': None,
                                     'metadata':{}}],
                'loading_info': {'table_type': table_type},
                'table_metadata': {'date_created': '11/16/20'}
                }

    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33, 57])},
                   table_metadata={'date_created': '11/16/20'},
                   column_descriptions={'signup_date': 'original signup date',
                                        'age': 'age of the user'},
                   column_metadata={'id': {'is_sorted': True},
                                    'age': {'interesting_values': [33, 57]}}
                   )

    description = dt.to_dictionary()

    assert description == expected


def test_unserializable_table(sample_df, tmpdir):
    dt = DataTable(sample_df, table_metadata={'not_serializable': sample_df['is_registered'].dtype})

    error = "DataTable is not json serializable: Object of type dtype is not JSON serializable"
    with pytest.raises(TypeError, match=error):
        dt.to_csv(str(tmpdir), encoding='utf-8', engine='python')


def test_serialize_wrong_format(sample_df, tmpdir):
    dt = DataTable(sample_df)

    error = 'must be one of the following formats: csv, pickle, parquet'
    with pytest.raises(ValueError, match=error):
        serialize.write_datatable(dt, str(tmpdir), format='test')


def test_to_csv(sample_df, tmpdir):
    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33, 57])},
                   column_descriptions={'signup_date': 'original signup date',
                                        'age': 'age of the user'},
                   column_metadata={'id': {'is_sorted': True},
                                    'age': {'interesting_values': [33, 57]}})

    dt.to_csv(str(tmpdir), encoding='utf-8', engine='python')
    _dt = deserialize.read_datatable(str(tmpdir))

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=_dt.index, sort_index=True),
                                  to_pandas(_dt.to_dataframe(), index=_dt.index, sort_index=True))
    assert dt == _dt


def test_to_csv_with_latlong(latlong_df, tmpdir):
    dt = DataTable(latlong_df, index='tuple_ints', logical_types={col: 'LatLong' for col in latlong_df.columns})
    dt.to_csv(str(tmpdir))
    _dt = deserialize.read_datatable(str(tmpdir))

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index, sort_index=True),
                                  to_pandas(_dt.to_dataframe(), index=_dt.index, sort_index=True))
    assert dt == _dt


def test_to_pickle(sample_df, tmpdir):
    dt = DataTable(sample_df)
    if not isinstance(sample_df, pd.DataFrame):
        msg = 'DataFrame type not compatible with pickle serialization. Please serialize to another format.'
        with pytest.raises(ValueError, match=msg):
            dt.to_pickle(str(tmpdir))
    else:
        dt.to_pickle(str(tmpdir))
        _dt = deserialize.read_datatable(str(tmpdir))

        pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index),
                                      to_pandas(_dt.to_dataframe(), index=_dt.index))
        assert dt == _dt


def test_to_pickle_with_latlong(latlong_df, tmpdir):
    dt = DataTable(latlong_df, logical_types={col: 'LatLong' for col in latlong_df.columns})
    if not isinstance(latlong_df, pd.DataFrame):
        msg = 'DataFrame type not compatible with pickle serialization. Please serialize to another format.'
        with pytest.raises(ValueError, match=msg):
            dt.to_pickle(str(tmpdir))
    else:
        dt.to_pickle(str(tmpdir))
        _dt = deserialize.read_datatable(str(tmpdir))

        pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index, sort_index=True),
                                      to_pandas(_dt.to_dataframe(), index=_dt.index, sort_index=True))
        assert dt == _dt


def test_to_parquet(sample_df, tmpdir):
    dt = DataTable(sample_df, index='id')
    dt.to_parquet(str(tmpdir))
    _dt = deserialize.read_datatable(str(tmpdir))
    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index, sort_index=True),
                                  to_pandas(_dt.to_dataframe(), index=_dt.index, sort_index=True))
    assert dt == _dt


def test_to_parquet_with_latlong(latlong_df, tmpdir):
    dt = DataTable(latlong_df, logical_types={col: 'LatLong' for col in latlong_df.columns})
    dt.to_parquet(str(tmpdir))
    _dt = deserialize.read_datatable(str(tmpdir))

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index, sort_index=True),
                                  to_pandas(_dt.to_dataframe(), index=_dt.index, sort_index=True))
    assert dt == _dt


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
    xfail_tmp_disappears(sample_df)

    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33, 57])})
    dt.to_csv(TEST_S3_URL, encoding='utf-8', engine='python')
    make_public(s3_client, s3_bucket)

    _dt = deserialize.read_datatable(TEST_S3_URL)

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert dt == _dt


def test_serialize_s3_pickle(sample_df_pandas, s3_client, s3_bucket):
    pandas_dt = DataTable(sample_df_pandas)
    pandas_dt.to_pickle(TEST_S3_URL)
    make_public(s3_client, s3_bucket)
    _dt = deserialize.read_datatable(TEST_S3_URL)

    pd.testing.assert_frame_equal(to_pandas(pandas_dt.to_dataframe(), index=pandas_dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert pandas_dt == _dt


def test_serialize_s3_parquet(sample_df, s3_client, s3_bucket):
    xfail_tmp_disappears(sample_df)

    dt = DataTable(sample_df)
    dt.to_parquet(TEST_S3_URL)
    make_public(s3_client, s3_bucket)
    _dt = deserialize.read_datatable(TEST_S3_URL)

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert dt == _dt


def test_to_csv_S3_anon(sample_df, s3_client, s3_bucket):
    xfail_tmp_disappears(sample_df)

    dt = DataTable(sample_df,
                   name='test_data',
                   index='id',
                   time_index='signup_date',
                   semantic_tags={'id': 'tag1'},
                   logical_types={'age': Ordinal(order=[25, 33, 57])})
    dt.to_csv(TEST_S3_URL, encoding='utf-8', engine='python', profile_name=False)
    make_public(s3_client, s3_bucket)

    _dt = deserialize.read_datatable(TEST_S3_URL, profile_name=False)

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert dt == _dt


def test_serialize_s3_pickle_anon(sample_df_pandas, s3_client, s3_bucket):
    pandas_dt = DataTable(sample_df_pandas)
    pandas_dt.to_pickle(TEST_S3_URL, profile_name=False)
    make_public(s3_client, s3_bucket)
    _dt = deserialize.read_datatable(TEST_S3_URL, profile_name=False)

    pd.testing.assert_frame_equal(to_pandas(pandas_dt.to_dataframe(), index=pandas_dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert pandas_dt == _dt


def test_serialize_s3_parquet_anon(sample_df, s3_client, s3_bucket):
    xfail_tmp_disappears(sample_df)

    dt = DataTable(sample_df)
    dt.to_parquet(TEST_S3_URL, profile_name=False)
    make_public(s3_client, s3_bucket)
    _dt = deserialize.read_datatable(TEST_S3_URL, profile_name=False)

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert dt == _dt


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
    except OSError:
        pass
    try:
        os.remove(test_path_config)
    except OSError:
        pass

    create_test_credentials(test_path)
    create_test_config(test_path_config)
    yield
    os.remove(test_path)
    os.remove(test_path_config)


def test_s3_test_profile(sample_df, s3_client, s3_bucket, setup_test_profile):
    xfail_tmp_disappears(sample_df)
    dt = DataTable(sample_df)
    dt.to_csv(TEST_S3_URL, encoding='utf-8', engine='python', profile_name='test')
    make_public(s3_client, s3_bucket)
    _dt = deserialize.read_datatable(TEST_S3_URL, profile_name='test')

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert dt == _dt


def test_serialize_url_csv(sample_df):
    dt = DataTable(sample_df)
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        dt.to_csv(URL, encoding='utf-8', engine='python')


def test_serialize_subdirs_not_removed(sample_df, tmpdir):
    dt = DataTable(sample_df)
    write_path = tmpdir.mkdir("test")
    test_dir = write_path.mkdir("test_dir")
    with open(str(write_path.join('table_description.json')), 'w') as f:
        json.dump('__SAMPLE_TEXT__', f)
    compression = None
    serialize.write_datatable(dt, path=str(write_path), index='1', sep='\t', encoding='utf-8', compression=compression)
    assert os.path.exists(str(test_dir))
    with open(str(write_path.join('table_description.json')), 'r') as f:
        assert '__SAMPLE_TEXT__' not in json.load(f)


def test_deserialize_url_csv(sample_df_pandas):
    dt = DataTable(sample_df_pandas, index='id')
    _dt = deserialize.read_datatable(URL)

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert dt == _dt


def test_deserialize_url_csv_anon(sample_df_pandas):
    dt = DataTable(sample_df_pandas, index='id')
    _dt = deserialize.read_datatable(URL, profile_name=False)

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert dt == _dt


def test_deserialize_s3_csv(sample_df_pandas):
    dt = DataTable(sample_df_pandas, index='id')
    _dt = deserialize.read_datatable(S3_URL)

    pd.testing.assert_frame_equal(to_pandas(dt.to_dataframe(), index=dt.index), to_pandas(_dt.to_dataframe(), index=_dt.index))
    assert dt == _dt


def test_check_later_schema_version():
    def test_version(major, minor, patch, raises=True):
        version_to_check = '.'.join([str(v) for v in [major, minor, patch]])
        if raises:
            warning_text = ('The schema version of the saved woodwork.DataTable '
                            '%s is greater than the latest supported %s. '
                            'You may need to upgrade woodwork. Attempting to load woodwork.DataTable ...'
                            % (version_to_check, serialize.SCHEMA_VERSION))
            with pytest.warns(UpgradeSchemaWarning, match=warning_text):
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
            with pytest.warns(OutdatedSchemaWarning, match=warning_text):
                deserialize._check_schema_version(version_to_check)
        else:
            with pytest.warns(None) as record:
                deserialize._check_schema_version(version_to_check)
            assert len(record) == 0

    major, minor, patch = [int(s) for s in serialize.SCHEMA_VERSION.split('.')]

    test_version(major - 1, minor, patch)
    test_version(major, minor - 1, patch, raises=False)
    test_version(major, minor, patch - 1, raises=False)
