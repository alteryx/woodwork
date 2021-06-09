import os

import pandas as pd
import pytest
from mock import patch

import woodwork as ww


def test_read_file_errors_no_content_type(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample')
    sample_df_pandas.to_csv(filepath, index=False)

    no_type_error = "Content type could not be inferred. Please specify content_type and try again."
    with pytest.raises(RuntimeError, match=no_type_error):
        ww.read_file(filepath=filepath)


def test_read_file_errors_unsupported(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample')
    sample_df_pandas.to_feather(filepath)

    content_type = "notacontenttype"
    not_supported_error = "Reading from content type {} is not currently supported".format(content_type)
    with pytest.raises(RuntimeError, match=not_supported_error):
        ww.read_file(filepath=filepath, content_type=content_type)


def test_read_file_uses_supplied_content_type(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample')
    sample_df_pandas.to_csv(filepath, index=False)

    df_from_csv = ww.read_file(filepath=filepath, content_type='csv')
    assert isinstance(df_from_csv.ww.schema, ww.table_schema.TableSchema)

    schema_df = sample_df_pandas.copy()
    # pandas does not read data into nullable types currently, so the types
    # in df_from_csv will be different than the types inferred from sample_df_pandas
    # which uses the nullable types
    schema_df = schema_df.astype({'age': 'float64', 'is_registered': 'object'})
    schema_df.ww.init()

    assert df_from_csv.ww.schema == schema_df.ww.schema
    pd.testing.assert_frame_equal(schema_df, df_from_csv)


def test_read_file_no_params(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.csv')
    sample_df_pandas.to_csv(filepath, index=False)

    df_from_csv = ww.read_file(filepath=filepath)
    assert isinstance(df_from_csv.ww.schema, ww.table_schema.TableSchema)

    schema_df = sample_df_pandas.copy()
    # pandas does not read data into nullable types currently, so the types
    # in df_from_csv will be different than the types inferred from sample_df_pandas
    # which uses the nullable types
    schema_df = schema_df.astype({'age': 'float64', 'is_registered': 'object'})
    schema_df.ww.init()

    assert df_from_csv.ww.schema == schema_df.ww.schema
    pd.testing.assert_frame_equal(schema_df, df_from_csv)


def test_read_file_with_woodwork_params(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.csv')
    sample_df_pandas.to_csv(filepath, index=False)
    logical_types = {
        'full_name': 'NaturalLanguage',
        'phone_number': 'PhoneNumber',
        'is_registered': 'BooleanNullable',
        'age': 'IntegerNullable'
    }
    semantic_tags = {
        'age': ['tag1', 'tag2'],
        'is_registered': ['tag3', 'tag4']
    }
    df_from_csv = ww.read_file(filepath=filepath,
                               index='id',
                               time_index='signup_date',
                               logical_types=logical_types,
                               semantic_tags=semantic_tags)
    assert isinstance(df_from_csv.ww.schema, ww.table_schema.TableSchema)

    schema_df = sample_df_pandas.copy()
    schema_df.ww.init(index='id',
                      time_index='signup_date',
                      logical_types=logical_types,
                      semantic_tags=semantic_tags)

    assert df_from_csv.ww.schema == schema_df.ww.schema
    pd.testing.assert_frame_equal(schema_df, df_from_csv)


def test_read_file_with_pandas_params(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.csv')
    sample_df_pandas.to_csv(filepath, index=False)
    nrows = 2

    df_from_csv = ww.read_file(filepath=filepath, nrows=nrows, dtype={'age': 'Int64', 'is_registered': 'boolean'})
    assert isinstance(df_from_csv.ww.schema, ww.table_schema.TableSchema)

    schema_df = sample_df_pandas.copy()
    schema_df.ww.init()

    assert df_from_csv.ww.schema == schema_df.ww.schema
    assert len(df_from_csv) == nrows
    pd.testing.assert_frame_equal(df_from_csv, schema_df.head(nrows))


@patch("woodwork.table_accessor._validate_accessor_params")
def test_read_file_validation_control(mock_validate_accessor_params, sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.csv')
    sample_df_pandas.to_csv(filepath, index=False)

    assert not mock_validate_accessor_params.called
    ww.read_file(filepath=filepath, validate=False)
    assert not mock_validate_accessor_params.called
    ww.read_file(filepath=filepath)
    assert mock_validate_accessor_params.called


def test_read_file_parquet(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.parquet')
    sample_df_pandas.to_parquet(filepath, index=False)

    content_types = ['parquet', 'application/parquet']
    for content_type in content_types:
        df_from_parquet = ww.read_file(filepath=filepath,
                                       content_type=content_type,
                                       index='id',
                                       use_nullable_dtypes=True)
        assert isinstance(df_from_parquet.ww.schema, ww.table_schema.TableSchema)

        schema_df = sample_df_pandas.copy()
        schema_df.ww.init(index='id')

        assert df_from_parquet.ww.schema == schema_df.ww.schema
        pd.testing.assert_frame_equal(df_from_parquet, schema_df)


def test_read_file_parquet_no_params(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.parquet')
    sample_df_pandas.to_parquet(filepath, index=False)

    df_from_parquet = ww.read_file(filepath=filepath)
    assert isinstance(df_from_parquet.ww.schema, ww.table_schema.TableSchema)

    schema_df = sample_df_pandas.copy()
    schema_df.ww.init()

    assert df_from_parquet.ww.schema == schema_df.ww.schema
    pd.testing.assert_frame_equal(df_from_parquet, schema_df)


def test_read_file_arrow(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.arrow')
    sample_df_pandas.to_feather(filepath)

    content_types = ['arrow', 'application/arrow']
    for content_type in content_types:
        df_from_arrow = ww.read_file(filepath=filepath,
                                     content_type=content_type,
                                     index='id')
        assert isinstance(df_from_arrow.ww.schema, ww.table_schema.TableSchema)

        schema_df = sample_df_pandas.copy()
        schema_df.ww.init(index='id')

        assert df_from_arrow.ww.schema == schema_df.ww.schema
        pd.testing.assert_frame_equal(df_from_arrow, schema_df)


def test_read_file_arrow_no_params(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.arrow')
    sample_df_pandas.to_feather(filepath)

    df_from_arrow = ww.read_file(filepath=filepath)
    assert isinstance(df_from_arrow.ww.schema, ww.table_schema.TableSchema)

    schema_df = sample_df_pandas.copy()
    schema_df.ww.init()

    assert df_from_arrow.ww.schema == schema_df.ww.schema
    pd.testing.assert_frame_equal(df_from_arrow, schema_df)


def test_read_file_feather(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.feather')
    sample_df_pandas.to_feather(filepath)

    content_types = ['feather', 'application/feather']
    for content_type in content_types:
        df_from_feather = ww.read_file(filepath=filepath,
                                       content_type=content_type,
                                       index='id')
        assert isinstance(df_from_feather.ww.schema, ww.table_schema.TableSchema)

        schema_df = sample_df_pandas.copy()
        schema_df.ww.init(index='id')

        assert df_from_feather.ww.schema == schema_df.ww.schema
        pd.testing.assert_frame_equal(df_from_feather, schema_df)


def test_read_file_feather_no_params(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.feather')
    sample_df_pandas.to_feather(filepath)

    df_from_feather = ww.read_file(filepath=filepath)
    assert isinstance(df_from_feather.ww.schema, ww.table_schema.TableSchema)

    schema_df = sample_df_pandas.copy()
    schema_df.ww.init()

    assert df_from_feather.ww.schema == schema_df.ww.schema
    pd.testing.assert_frame_equal(df_from_feather, schema_df)
