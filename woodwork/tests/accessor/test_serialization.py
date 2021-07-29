import json
import os

import boto3
import pandas as pd
import pytest
from mock import patch

import woodwork.deserialize as deserialize
import woodwork.serialize as serialize
from woodwork.accessor_utils import _is_dask_dataframe, _is_koalas_dataframe
from woodwork.exceptions import (
    OutdatedSchemaWarning,
    UpgradeSchemaWarning,
    WoodworkNotInitError
)
from woodwork.logical_types import Categorical, Ordinal
from woodwork.tests.testing_utils import to_pandas

BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_serialization_woodwork_table_schema_{}.tar".format(serialize.SCHEMA_VERSION)
S3_URL = "s3://woodwork-static/" + TEST_FILE
URL = "https://woodwork-static.s3.amazonaws.com/" + TEST_FILE
TEST_KEY = "test_access_key_es"


def xfail_tmp_disappears(dataframe):
    # TODO: tmp file disappears after deserialize step, cannot check equality with Dask
    if not isinstance(dataframe, pd.DataFrame):
        pytest.xfail('tmp file disappears after deserialize step, cannot check equality with Dask')


def test_error_before_table_init(sample_df, tmpdir):
    error_message = "Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init"

    with pytest.raises(WoodworkNotInitError, match=error_message):
        sample_df.ww.to_dictionary()

    with pytest.raises(WoodworkNotInitError, match=error_message):
        sample_df.ww.to_disk(str(tmpdir), format='csv')


def test_to_dictionary(sample_df):
    if _is_dask_dataframe(sample_df):
        table_type = 'dask'
        age_cat_type_dict = {
            'type': 'category',
            'cat_values': [33, 57],
            'cat_dtype': 'int64'
        }
        cat_type_dict = {
            'type': 'category',
            'cat_values': ['a', 'b', 'c'],
            'cat_dtype': 'object'
        }
    elif _is_koalas_dataframe(sample_df):
        table_type = 'koalas'
        age_cat_type_dict = {
            'type': 'string'
        }
        cat_type_dict = {
            'type': 'string'
        }
    else:
        table_type = 'pandas'
        age_cat_type_dict = {
            'type': 'category',
            'cat_values': [33, 57],
            'cat_dtype': 'object'
        }
        cat_type_dict = {
            'type': 'category',
            'cat_values': ['a', 'b', 'c'],
            'cat_dtype': 'object'
        }

    int_val = 'int64'
    nullable_int_val = 'Int64'
    string_val = 'string'
    bool_val = 'boolean'
    double_val = 'float64'

    expected = {'schema_version': serialize.SCHEMA_VERSION,
                'name': 'test_data',
                'index': 'id',
                'time_index': None,
                'column_typing_info': [{'name': 'id',
                                        'ordinal': 0,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'Integer'},
                                        'physical_type': {'type': int_val},
                                        'semantic_tags': ['index', 'tag1'],
                                        'description': None,
                                        'origin': None,
                                        'metadata':{'is_sorted': True}},
                                       {'name': 'full_name',
                                        'ordinal': 1,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'Unknown'},
                                        'physical_type': {'type': string_val},
                                        'semantic_tags': [],
                                        'description': None,
                                        'origin': None,
                                        'metadata':{}},
                                       {'name': 'email',
                                        'ordinal': 2,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'EmailAddress'},
                                        'physical_type': {'type': string_val},
                                        'semantic_tags': [],
                                        'description': None,
                                        'origin': None,
                                        'metadata':{}},
                                       {'name': 'phone_number',
                                        'ordinal': 3,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'Unknown'},
                                        'physical_type': {'type': string_val},
                                        'semantic_tags': [],
                                        'description': None,
                                        'origin': 'base',
                                        'metadata': {}},
                                       {'name': 'age',
                                        'ordinal': 4,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {'order': [25, 33, 57]}, 'type': 'Ordinal'},
                                        'physical_type': age_cat_type_dict,
                                        'semantic_tags': ['category'],
                                        'description': 'age of the user',
                                        'origin': 'base',
                                        'metadata':{'interesting_values': [33, 57]}},
                                       {'name': 'signup_date',
                                        'ordinal': 5,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {'datetime_format': None},
                                                         'type': 'Datetime'},
                                        'physical_type': {'type': 'datetime64[ns]'},
                                        'semantic_tags': [],
                                        'description': 'original signup date',
                                        'origin': 'engineered',
                                        'metadata':{}},
                                       {'name': 'is_registered',
                                        'ordinal': 6,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'BooleanNullable'},
                                        'physical_type': {'type': bool_val},
                                        'semantic_tags': [],
                                        'description': None,
                                        'origin': None,
                                        'metadata': {}},
                                       {'name': 'double',
                                        'ordinal': 7,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'Double'},
                                        'physical_type': {'type': double_val},
                                        'semantic_tags': ['numeric'],
                                        'description': None,
                                        'origin': None,
                                        'metadata': {}},
                                       {'name': 'double_with_nan',
                                        'ordinal': 8,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'Double'},
                                        'physical_type': {'type': double_val},
                                        'semantic_tags': ['numeric'],
                                        'description': None,
                                        'origin': None,
                                        'metadata': {}},
                                       {'name': 'integer',
                                        'ordinal': 9,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'Integer'},
                                        'physical_type': {'type': int_val},
                                        'semantic_tags': ['numeric'],
                                        'description': None,
                                        'origin': None,
                                        'metadata': {}},
                                       {'name': 'nullable_integer',
                                        'ordinal': 10,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'IntegerNullable'},
                                        'physical_type': {'type': nullable_int_val},
                                        'semantic_tags': ['numeric'],
                                        'description': None,
                                        'origin': None,
                                        'metadata': {}},
                                       {'name': 'boolean',
                                        'ordinal': 11,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'Boolean'},
                                        'physical_type': {'type': 'bool'},
                                        'semantic_tags': [],
                                        'description': None,
                                        'origin': None,
                                        'metadata': {}},
                                       {'name': 'categorical',
                                        'ordinal': 12,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {}, 'type': 'Categorical'},
                                        'physical_type': cat_type_dict,
                                        'semantic_tags': ['category'],
                                        'description': None,
                                        'origin': None,
                                        'metadata': {}},
                                       {'name': 'datetime_with_NaT',
                                        'ordinal': 13,
                                        'use_standard_tags': True,
                                        'logical_type': {'parameters': {'datetime_format': None}, 'type': 'Datetime'},
                                        'physical_type': {'type': 'datetime64[ns]'},
                                        'semantic_tags': [],
                                        'description': None,
                                        'origin': None,
                                        'metadata':{}}],
                'loading_info': {'table_type': table_type},
                'table_metadata': {'date_created': '11/16/20'}
                }
    sample_df.ww.init(
        name='test_data',
        index='id',
        semantic_tags={'id': 'tag1'},
        logical_types={'age': Ordinal(order=[25, 33, 57])},
        table_metadata={'date_created': '11/16/20'},
        column_descriptions={'signup_date': 'original signup date',
                             'age': 'age of the user'},
        column_origins={'phone_number': 'base',
                        'age': 'base',
                        'signup_date': 'engineered'},
        column_metadata={'id': {'is_sorted': True},
                         'age': {'interesting_values': [33, 57]}}
    )

    description = sample_df.ww.to_dictionary()
    assert description == expected


def test_unserializable_table(sample_df, tmpdir):
    sample_df.ww.init(table_metadata={'not_serializable': sample_df['is_registered'].dtype})

    error = "Woodwork table is not json serializable. Check table and column metadata for values that may not be serializable."
    with pytest.raises(TypeError, match=error):
        sample_df.ww.to_disk(str(tmpdir), format='csv', encoding='utf-8', engine='python')


def test_serialize_wrong_format(sample_df, tmpdir):
    sample_df.ww.init()

    error = 'must be one of the following formats: csv, pickle, parquet'
    with pytest.raises(ValueError, match=error):
        serialize.write_woodwork_table(sample_df, str(tmpdir), format='test')


def test_to_csv(sample_df, tmpdir):
    if _is_dask_dataframe(sample_df):
        # Dask errors with pd.NA in some partitions, but not others
        sample_df['age'] = sample_df['age'].fillna(25)
    sample_df.ww.init(
        name='test_data',
        index='id',
        semantic_tags={'id': 'tag1'},
        logical_types={'age': Ordinal(order=[25, 33, 57])},
        column_descriptions={'signup_date': 'original signup date',
                             'age': 'age of the user'},
        column_origins={'phone_number': 'base',
                        'age': 'base',
                        'signup_date': 'engineered'},
        column_metadata={'id': {'is_sorted': True},
                         'age': {'interesting_values': [33, 57]}})
    sample_df.ww.to_disk(str(tmpdir), format='csv', encoding='utf-8', engine='python')
    deserialized_df = deserialize.read_woodwork_table(str(tmpdir))

    pd.testing.assert_frame_equal(to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True),
                                  to_pandas(sample_df, index=sample_df.ww.index, sort_index=True))
    assert deserialized_df.ww.schema == sample_df.ww.schema


def test_to_csv_with_latlong(latlong_df, tmpdir):
    latlong_df.ww.init(index='tuple_ints', logical_types={col: 'LatLong' for col in latlong_df.columns})
    latlong_df.ww.to_disk(str(tmpdir))
    deserialized_df = deserialize.read_woodwork_table(str(tmpdir))

    pd.testing.assert_frame_equal(to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True),
                                  to_pandas(latlong_df, index=latlong_df.ww.index, sort_index=True))
    assert deserialized_df.ww.schema == latlong_df.ww.schema


def test_to_disk_with_whitespace(whitespace_df, tmpdir):
    formats = ['csv', 'parquet', 'pickle']
    for format in formats:
        df = whitespace_df.copy()
        df.ww.init(index='id', logical_types={'comments': 'NaturalLanguage'})
        if format == 'pickle' and not isinstance(df, pd.DataFrame):
            msg = 'DataFrame type not compatible with pickle serialization. Please serialize to another format.'
            with pytest.raises(ValueError, match=msg):
                df.ww.to_disk(str(tmpdir), format='pickle')
        else:
            df.ww.to_disk(str(tmpdir), format=format)
            deserialized_df = deserialize.read_woodwork_table(str(tmpdir))
            assert deserialized_df.ww.schema == df.ww.schema
            pd.testing.assert_frame_equal(to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True),
                                          to_pandas(df, index=df.ww.index, sort_index=True))


def test_to_csv_use_standard_tags(sample_df, tmpdir):
    no_standard_tags_df = sample_df.copy()
    no_standard_tags_df.ww.init(use_standard_tags=False)

    no_standard_tags_df.ww.to_disk(str(tmpdir), format='csv', encoding='utf-8', engine='python')
    deserialized_no_tags_df = deserialize.read_woodwork_table(str(tmpdir))

    standard_tags_df = sample_df.copy()
    standard_tags_df.ww.init(use_standard_tags=True)

    standard_tags_df.ww.to_disk(str(tmpdir), format='csv', encoding='utf-8', engine='python')
    deserialized_tags_df = deserialize.read_woodwork_table(str(tmpdir))

    assert no_standard_tags_df.ww.schema != standard_tags_df.ww.schema

    assert deserialized_no_tags_df.ww.schema == no_standard_tags_df.ww.schema
    assert deserialized_tags_df.ww.schema == standard_tags_df.ww.schema


def test_to_pickle(sample_df, tmpdir):
    sample_df.ww.init()
    if not isinstance(sample_df, pd.DataFrame):
        msg = 'DataFrame type not compatible with pickle serialization. Please serialize to another format.'
        with pytest.raises(ValueError, match=msg):
            sample_df.ww.to_disk(str(tmpdir), format='pickle')
    else:
        sample_df.ww.to_disk(str(tmpdir), format='pickle')
        deserialized_df = deserialize.read_woodwork_table(str(tmpdir))

        pd.testing.assert_frame_equal(to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True),
                                      to_pandas(sample_df, index=sample_df.ww.index, sort_index=True))
        assert deserialized_df.ww.schema == sample_df.ww.schema


def test_to_pickle_with_latlong(latlong_df, tmpdir):
    latlong_df.ww.init(logical_types={col: 'LatLong' for col in latlong_df.columns})
    if not isinstance(latlong_df, pd.DataFrame):
        msg = 'DataFrame type not compatible with pickle serialization. Please serialize to another format.'
        with pytest.raises(ValueError, match=msg):
            latlong_df.ww.to_disk(str(tmpdir), format='pickle')
    else:
        latlong_df.ww.to_disk(str(tmpdir), format='pickle')
        deserialized_df = deserialize.read_woodwork_table(str(tmpdir))

        pd.testing.assert_frame_equal(to_pandas(latlong_df, index=latlong_df.ww.index, sort_index=True),
                                      to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
        assert latlong_df.ww.schema == deserialized_df.ww.schema


def test_to_parquet(sample_df, tmpdir):
    sample_df.ww.init(index='id')
    sample_df.ww.to_disk(str(tmpdir), format='parquet')
    deserialized_df = deserialize.read_woodwork_table(str(tmpdir))
    pd.testing.assert_frame_equal(to_pandas(sample_df, index=sample_df.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert sample_df.ww.schema == deserialized_df.ww.schema


def test_to_parquet_with_latlong(latlong_df, tmpdir):
    latlong_df.ww.init(logical_types={col: 'LatLong' for col in latlong_df.columns})
    latlong_df.ww.to_disk(str(tmpdir), format='parquet')
    deserialized_df = deserialize.read_woodwork_table(str(tmpdir))

    pd.testing.assert_frame_equal(to_pandas(latlong_df, index=latlong_df.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert latlong_df.ww.schema == deserialized_df.ww.schema


def test_to_arrow(sample_df, tmpdir):
    if not isinstance(sample_df, pd.DataFrame):
        pytest.xfail('Arrow IPC format (Feather) not supported on Dask or Koalas')
    sample_df.ww.init(index='id')
    sample_df.ww.to_disk(str(tmpdir), format='arrow')
    deserialized_df = deserialize.read_woodwork_table(str(tmpdir))
    pd.testing.assert_frame_equal(to_pandas(sample_df, index=sample_df.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert sample_df.ww.schema == deserialized_df.ww.schema


def test_to_arrow_with_latlong(latlong_df, tmpdir):
    if not isinstance(latlong_df, pd.DataFrame):
        pytest.xfail('Arrow IPC format (Feather) not supported on Dask or Koalas')
    latlong_df.ww.init(logical_types={col: 'LatLong' for col in latlong_df.columns})
    latlong_df.ww.to_disk(str(tmpdir), format='arrow')
    deserialized_df = deserialize.read_woodwork_table(str(tmpdir))

    pd.testing.assert_frame_equal(to_pandas(latlong_df, index=latlong_df.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert latlong_df.ww.schema == deserialized_df.ww.schema


def test_to_feather(sample_df, tmpdir):
    if not isinstance(sample_df, pd.DataFrame):
        pytest.xfail('Arrow IPC format (Feather) not supported on Dask or Koalas')
    sample_df.ww.init(index='id')
    sample_df.ww.to_disk(str(tmpdir), format='feather')
    deserialized_df = deserialize.read_woodwork_table(str(tmpdir))
    pd.testing.assert_frame_equal(to_pandas(sample_df, index=sample_df.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert sample_df.ww.schema == deserialized_df.ww.schema


def test_to_feather_with_latlong(latlong_df, tmpdir):
    if not isinstance(latlong_df, pd.DataFrame):
        pytest.xfail('Arrow IPC format (Feather) not supported on Dask or Koalas')
    latlong_df.ww.init(logical_types={col: 'LatLong' for col in latlong_df.columns})
    latlong_df.ww.to_disk(str(tmpdir), format='feather')
    deserialized_df = deserialize.read_woodwork_table(str(tmpdir))

    pd.testing.assert_frame_equal(to_pandas(latlong_df, index=latlong_df.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert latlong_df.ww.schema == deserialized_df.ww.schema


def test_to_orc(sample_df, tmpdir):
    sample_df.ww.init(index='id')
    if _is_dask_dataframe(sample_df):
        msg = 'DataFrame type not compatible with orc serialization. Please serialize to another format.'
        with pytest.raises(ValueError, match=msg):
            sample_df.ww.to_disk(str(tmpdir), format='orc')
    else:
        sample_df.ww.to_disk(str(tmpdir), format='orc')
        deserialized_df = deserialize.read_woodwork_table(str(tmpdir))
        pd.testing.assert_frame_equal(to_pandas(sample_df, index=sample_df.ww.index, sort_index=True),
                                      to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
        assert sample_df.ww.schema == deserialized_df.ww.schema


def test_to_orc_with_latlong(latlong_df, tmpdir):
    latlong_df.ww.init(logical_types={col: 'LatLong' for col in latlong_df.columns})
    if _is_dask_dataframe(latlong_df):
        msg = 'DataFrame type not compatible with orc serialization. Please serialize to another format.'
        with pytest.raises(ValueError, match=msg):
            latlong_df.ww.to_disk(str(tmpdir), format='orc')
    else:
        latlong_df.ww.to_disk(str(tmpdir), format='orc')
        deserialized_df = deserialize.read_woodwork_table(str(tmpdir))

        pd.testing.assert_frame_equal(to_pandas(latlong_df, index=latlong_df.ww.index, sort_index=True),
                                      to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
        assert latlong_df.ww.schema == deserialized_df.ww.schema


def test_categorical_dtype_serialization(serialize_df, tmpdir):
    ltypes = {
        'cat_int': Categorical,
        'ord_int': Ordinal(order=[1, 2]),
        'cat_float': Categorical,
        'ord_float': Ordinal(order=[1.0, 2.0]),
        'cat_bool': Categorical,
        'ord_bool': Ordinal(order=[True, False]),
    }
    if isinstance(serialize_df, pd.DataFrame):
        formats = ['csv', 'pickle', 'parquet']
    else:
        formats = ['csv']

    for format in formats:
        df = serialize_df.copy()
        df.ww.init(index='id', logical_types=ltypes)
        df.ww.to_disk(str(tmpdir), format=format)
        deserialized_df = deserialize.read_woodwork_table(str(tmpdir))
        pd.testing.assert_frame_equal(to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True),
                                      to_pandas(df, index=df.ww.index, sort_index=True))
        assert deserialized_df.ww.schema == df.ww.schema


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

    sample_df.ww.init(
        name='test_data',
        index='id',
        semantic_tags={'id': 'tag1'},
        logical_types={'age': Ordinal(order=[25, 33, 57])})
    sample_df.ww.to_disk(TEST_S3_URL, format='csv', encoding='utf-8', engine='python')
    make_public(s3_client, s3_bucket)

    deserialized_df = deserialize.read_woodwork_table(TEST_S3_URL)

    pd.testing.assert_frame_equal(to_pandas(sample_df, index=sample_df.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert sample_df.ww.schema == deserialized_df.ww.schema


def test_serialize_s3_pickle(sample_df_pandas, s3_client, s3_bucket):
    sample_df_pandas.ww.init()
    sample_df_pandas.ww.to_disk(TEST_S3_URL, format='pickle')
    make_public(s3_client, s3_bucket)
    deserialized_df = deserialize.read_woodwork_table(TEST_S3_URL)

    pd.testing.assert_frame_equal(to_pandas(sample_df_pandas, index=sample_df_pandas.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert sample_df_pandas.ww.schema == deserialized_df.ww.schema


def test_serialize_s3_parquet(sample_df, s3_client, s3_bucket):
    xfail_tmp_disappears(sample_df)

    sample_df.ww.init()
    sample_df.ww.to_disk(TEST_S3_URL, format='parquet')
    make_public(s3_client, s3_bucket)
    deserialized_df = deserialize.read_woodwork_table(TEST_S3_URL)

    pd.testing.assert_frame_equal(to_pandas(sample_df, index=sample_df.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert sample_df.ww.schema == deserialized_df.ww.schema


def test_to_csv_S3_anon(sample_df, s3_client, s3_bucket):
    xfail_tmp_disappears(sample_df)

    sample_df.ww.init(
        name='test_data',
        index='id',
        time_index='signup_date',
        semantic_tags={'id': 'tag1'},
        logical_types={'age': Ordinal(order=[25, 33, 57])})
    sample_df.ww.to_disk(TEST_S3_URL, format='csv', encoding='utf-8', engine='python', profile_name=False)
    make_public(s3_client, s3_bucket)

    deserialized_df = deserialize.read_woodwork_table(TEST_S3_URL, profile_name=False)

    pd.testing.assert_frame_equal(to_pandas(sample_df, index=sample_df.ww.index, sort_index=True),
                                  to_pandas(deserialized_df, index=deserialized_df.ww.index, sort_index=True))
    assert sample_df.ww.schema == deserialized_df.ww.schema


def test_serialize_s3_pickle_anon(sample_df_pandas, s3_client, s3_bucket):
    sample_df_pandas.ww.init()
    sample_df_pandas.ww.to_disk(TEST_S3_URL, format='pickle', profile_name=False)
    make_public(s3_client, s3_bucket)
    deserialized_df = deserialize.read_woodwork_table(TEST_S3_URL, profile_name=False)

    pd.testing.assert_frame_equal(to_pandas(sample_df_pandas, index=sample_df_pandas.ww.index), to_pandas(deserialized_df, index=deserialized_df.ww.index))
    assert sample_df_pandas.ww.schema == deserialized_df.ww.schema


def test_serialize_s3_parquet_anon(sample_df, s3_client, s3_bucket):
    xfail_tmp_disappears(sample_df)

    sample_df.ww.init()
    sample_df.ww.to_disk(TEST_S3_URL, format='parquet', profile_name=False)
    make_public(s3_client, s3_bucket)
    deserialized_df = deserialize.read_woodwork_table(TEST_S3_URL, profile_name=False)

    pd.testing.assert_frame_equal(to_pandas(sample_df, index=sample_df.ww.index), to_pandas(deserialized_df, index=deserialized_df.ww.index))
    assert sample_df.ww.schema == deserialized_df.ww.schema


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
    sample_df.ww.init()
    sample_df.ww.to_disk(TEST_S3_URL, format='csv', encoding='utf-8', engine='python', profile_name='test')
    make_public(s3_client, s3_bucket)
    deserialized_df = deserialize.read_woodwork_table(TEST_S3_URL, profile_name='test')

    pd.testing.assert_frame_equal(to_pandas(sample_df, index=sample_df.ww.index), to_pandas(deserialized_df, index=deserialized_df.ww.index))
    assert sample_df.ww.schema == deserialized_df.ww.schema


def test_serialize_url_csv(sample_df):
    sample_df.ww.init()
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        sample_df.ww.to_disk(URL, format='csv', encoding='utf-8', engine='python')


def test_serialize_subdirs_not_removed(sample_df, tmpdir):
    sample_df.ww.init()
    write_path = tmpdir.mkdir("test")
    test_dir = write_path.mkdir("test_dir")
    with open(str(write_path.join('woodwork_typing_info.json')), 'w') as f:
        json.dump('__SAMPLE_TEXT__', f)
    compression = None
    serialize.write_woodwork_table(sample_df, path=str(write_path), index='1', sep='\t', encoding='utf-8', compression=compression)
    assert os.path.exists(str(test_dir))
    with open(str(write_path.join('woodwork_typing_info.json')), 'r') as f:
        assert '__SAMPLE_TEXT__' not in json.load(f)


def test_deserialize_url_csv(sample_df_pandas):
    sample_df_pandas.ww.init(index='id')
    deserialized_df = deserialize.read_woodwork_table(URL)
    pd.testing.assert_frame_equal(to_pandas(sample_df_pandas, index=sample_df_pandas.ww.index), to_pandas(deserialized_df, index=deserialized_df.ww.index))
    assert sample_df_pandas.ww.schema == deserialized_df.ww.schema


def test_deserialize_url_csv_anon(sample_df_pandas):
    sample_df_pandas.ww.init(index='id')
    deserialized_df = deserialize.read_woodwork_table(URL, profile_name=False)

    pd.testing.assert_frame_equal(to_pandas(sample_df_pandas, index=sample_df_pandas.ww.index), to_pandas(deserialized_df, index=deserialized_df.ww.index))
    assert sample_df_pandas.ww.schema == deserialized_df.ww.schema


def test_deserialize_s3_csv(sample_df_pandas):
    sample_df_pandas.ww.init(index='id')
    deserialized_df = deserialize.read_woodwork_table(S3_URL)

    pd.testing.assert_frame_equal(to_pandas(sample_df_pandas, index=sample_df_pandas.ww.index), to_pandas(deserialized_df, index=deserialized_df.ww.index))
    assert sample_df_pandas.ww.schema == deserialized_df.ww.schema


@patch("woodwork.table_accessor._validate_accessor_params")
def test_deserialize_validation_control(mock_validate_accessor_params):
    assert not mock_validate_accessor_params.called
    deserialize.read_woodwork_table(URL)
    assert not mock_validate_accessor_params.called
    deserialize.read_woodwork_table(URL, validate=True)
    assert mock_validate_accessor_params.called


def test_check_later_schema_version():
    def test_version(major, minor, patch, raises=True):
        version_to_check = '.'.join([str(v) for v in [major, minor, patch]])
        if raises:
            warning_text = ('The schema version of the saved Woodwork table '
                            '%s is greater than the latest supported %s. '
                            'You may need to upgrade woodwork. Attempting to load Woodwork table ...'
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
            warning_text = ('The schema version of the saved Woodwork table '
                            '%s is no longer supported by this version '
                            'of woodwork. Attempting to load Woodwork table ...'
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
