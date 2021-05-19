
import os
import re

import numpy as np
import pandas as pd
import pytest
from mock import patch

import woodwork as ww
from woodwork.logical_types import (
    Age,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    Integer,
    IntegerNullable,
    NaturalLanguage,
    Ordinal,
    PostalCode,
    SubRegionCode
)
from woodwork.type_sys.utils import (
    _get_specified_ltype_params,
    _is_numeric_series,
    list_logical_types,
    list_semantic_tags
)
from woodwork.utils import (
    _convert_input_to_set,
    _get_column_logical_type,
    _is_null_latlong,
    _is_s3,
    _is_url,
    _is_valid_latlong_series,
    _is_valid_latlong_value,
    _parse_logical_type,
    _reformat_to_latlong,
    _to_latlong_float,
    camel_to_snake,
    concat,
    get_valid_mi_types,
    import_or_none,
    import_or_raise
)
from woodwork.tests.testing_utils import to_pandas

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_camel_to_snake():
    test_items = {
        'PostalCode': 'postal_code',
        'SubRegionCode': 'sub_region_code',
        'NaturalLanguage': 'natural_language',
        'Categorical': 'categorical',
    }

    for key, value in test_items.items():
        assert camel_to_snake(key) == value


def test_convert_input_to_set():
    error_message = "semantic_tags must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _convert_input_to_set(int)

    error_message = "test_text must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _convert_input_to_set({'index': {}, 'time_index': {}}, 'test_text')

    error_message = "include parameter must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        _convert_input_to_set(['index', 1], 'include parameter')

    semantic_tags_from_single = _convert_input_to_set('index', 'include parameter')
    assert semantic_tags_from_single == {'index'}

    semantic_tags_from_list = _convert_input_to_set(['index', 'numeric', 'category'])
    assert semantic_tags_from_list == {'index', 'numeric', 'category'}

    semantic_tags_from_set = _convert_input_to_set({'index', 'numeric', 'category'}, 'include parameter')
    assert semantic_tags_from_set == {'index', 'numeric', 'category'}


@patch("woodwork.utils._validate_string_tags")
@patch("woodwork.utils._validate_tags_input_type")
def test_validation_methods_called(mock_validate_input_type, mock_validate_strings):
    assert not mock_validate_input_type.called
    assert not mock_validate_strings.called

    _convert_input_to_set('test_tag', validate=False)
    assert not mock_validate_input_type.called

    _convert_input_to_set('test_tag', validate=True)
    assert mock_validate_input_type.called

    _convert_input_to_set(['test_tag', 'tag2'], validate=False)
    assert not mock_validate_strings.called

    _convert_input_to_set(['test_tag', 'tag2'], validate=True)
    assert mock_validate_strings.called


def test_list_logical_types_default():
    all_ltypes = ww.logical_types.LogicalType.__subclasses__()

    df = list_logical_types()

    assert set(df.columns) == {'name', 'type_string', 'description', 'physical_type',
                               'standard_tags', 'is_default_type', 'is_registered', 'parent_type'}

    assert len(all_ltypes) == len(df)
    for name in df['name']:
        assert ww.type_system.str_to_logical_type(name) in all_ltypes
    assert all(df['is_default_type'])
    assert all(df['is_registered'])


def test_list_logical_types_customized_type_system():
    ww.type_system.remove_type('URL')

    class CustomRegistered(ww.logical_types.LogicalType):
        primary_dtype = 'int64'

    class CustomNotRegistered(ww.logical_types.LogicalType):
        primary_dtype = 'int64'

    ww.type_system.add_type(CustomRegistered)
    all_ltypes = ww.logical_types.LogicalType.__subclasses__()
    df = list_logical_types()
    assert len(all_ltypes) == len(df)
    # Check that URL is unregistered
    url = df[df.name == 'URL'].iloc[0]
    assert url.is_default_type
    assert not url.is_registered

    # Check that new registered type is present and shows as registered
    index = df.name == 'CustomRegistered'
    assert index.any()
    custom = df[index].iloc[0]
    assert not custom.is_default_type
    assert custom.is_registered

    # Check that new unregistered type is present and shows as not registered
    index = df.name == 'CustomNotRegistered'
    assert index.any()
    custom = df[index].iloc[0]
    assert not custom.is_default_type
    assert not custom.is_registered
    ww.type_system.reset_defaults()


def test_list_semantic_tags():
    df = list_semantic_tags()

    assert set(df.columns) == {'name', 'is_standard_tag', 'valid_logical_types'}

    for name, log_type_list in df[['name', 'valid_logical_types']].values:
        if name not in ['index', 'time_index', 'date_of_birth']:
            for log_type in log_type_list:
                assert name in log_type.standard_tags


def test_read_file_errors_no_content_type(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample')
    sample_df_pandas.to_csv(filepath, index=False)

    no_type_error = "Content type could not be inferred. Please specify content_type and try again."
    with pytest.raises(RuntimeError, match=no_type_error):
        ww.read_file(filepath=filepath)


def test_read_file_errors_unsupported(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample')
    sample_df_pandas.to_feather(filepath)

    content_type = "application/feather"
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


def test_is_numeric_datetime_series(time_index_df):
    assert _is_numeric_series(time_index_df['ints'], None)
    assert _is_numeric_series(time_index_df['ints'], Double)
    assert not _is_numeric_series(time_index_df['ints'], Categorical)
    assert _is_numeric_series(time_index_df['ints'], Datetime)

    assert not _is_numeric_series(time_index_df['strs'], None)
    assert not _is_numeric_series(time_index_df['strs'], 'Categorical')
    assert not _is_numeric_series(time_index_df['strs'], Categorical)
    assert _is_numeric_series(time_index_df['strs'], Double)
    assert _is_numeric_series(time_index_df['strs'], 'Double')

    assert not _is_numeric_series(time_index_df['bools'], None)
    assert not _is_numeric_series(time_index_df['bools'], 'Boolean')

    assert not _is_numeric_series(time_index_df['times'], None)
    assert not _is_numeric_series(time_index_df['times'], Datetime)

    assert not _is_numeric_series(time_index_df['letters'], None)
    assert not _is_numeric_series(time_index_df['letters'], Double)
    assert not _is_numeric_series(time_index_df['letters'], Categorical)


def test_get_ltype_params():
    params_empty_class = _get_specified_ltype_params(Categorical)
    assert params_empty_class == {}
    params_empty = _get_specified_ltype_params(Categorical())
    assert params_empty == {}

    params_class = _get_specified_ltype_params(Datetime)
    assert params_class == {}

    params_null = _get_specified_ltype_params(Datetime())
    assert params_null == {'datetime_format': None}

    ymd = '%Y-%m-%d'
    params_value = _get_specified_ltype_params(Datetime(datetime_format=ymd))
    assert params_value == {'datetime_format': ymd}


def test_import_or_raise():
    assert import_or_raise('pandas', 'Module pandas could not be found') == pd

    error = 'Module nonexistent could not be found.'
    with pytest.raises(ImportError, match=error):
        import_or_raise('nonexistent', error)


def test_import_or_none():
    assert import_or_none('pandas') == pd
    assert import_or_none('nonexistent') is None


def test_is_url():
    assert _is_url('https://www.google.com/')
    assert not _is_url('google.com')


def test_is_s3():
    assert _is_s3('s3://test-bucket/test-key')
    assert not _is_s3('https://woodwork-static.s3.amazonaws.com/')


def test_reformat_to_latlong_errors():
    for latlong in [{1, 2, 3}, '{1, 2, 3}', 'This is text']:
        error = (f'LatLongs must either be a tuple, a list, or a string representation of a tuple. {latlong} does not fit the criteria.')
        with pytest.raises(ValueError, match=error):
            _reformat_to_latlong(latlong)

    error = re.escape("LatLongs must either be a tuple, a list, or a string representation of a tuple. (1,2) does not fit the criteria.")
    with pytest.raises(ValueError, match=error):
        _reformat_to_latlong("'(1,2)'")

    for latlong in [(1, 2, 3), '(1, 2, 3)']:
        error = re.escape("LatLong values must have exactly two values. (1, 2, 3) does not have two values.")
        with pytest.raises(ValueError, match=error):
            _reformat_to_latlong(latlong)

    error = re.escape("Latitude and Longitude values must be in decimal degrees. The latitude or longitude represented by 41deg52\'54\" N cannot be converted to a float.")
    with pytest.raises(ValueError, match=error):
        _reformat_to_latlong(('41deg52\'54\" N', '21deg22\'54\" W'))


def test_reformat_to_latlong():
    simple_latlong = (1, 2)

    assert _reformat_to_latlong((1, 2)) == simple_latlong
    assert _reformat_to_latlong(('1', '2')) == simple_latlong
    assert _reformat_to_latlong('(1,2)') == simple_latlong

    # Check non-standard tuple formats
    assert _reformat_to_latlong([1, 2]) == simple_latlong
    assert _reformat_to_latlong(['1', '2']) == simple_latlong
    assert _reformat_to_latlong('[1, 2]') == simple_latlong
    assert _reformat_to_latlong('1, 2') == simple_latlong

    assert _reformat_to_latlong(None) is np.nan
    assert _reformat_to_latlong((1, np.nan)) == (1, np.nan)
    assert _reformat_to_latlong((np.nan, '1')) == (np.nan, 1)

    # This is how csv and parquet will deserialize
    assert _reformat_to_latlong('(1, nan)') == (1, np.nan)
    assert _reformat_to_latlong('(NaN, 9)') == (np.nan, 9)


def test_reformat_to_latlong_list():
    simple_latlong = [1, 2]

    assert _reformat_to_latlong((1, 2), use_list=True) == simple_latlong
    assert _reformat_to_latlong(('1', '2'), use_list=True) == simple_latlong
    assert _reformat_to_latlong('(1,2)', use_list=True) == simple_latlong

    assert _reformat_to_latlong([1, 2], use_list=True) == simple_latlong
    assert _reformat_to_latlong(['1', '2'], use_list=True) == simple_latlong
    assert _reformat_to_latlong('[1, 2]', use_list=True) == simple_latlong
    assert _reformat_to_latlong('1, 2', use_list=True) == simple_latlong

    assert _reformat_to_latlong((1, np.nan), use_list=True) == [1, np.nan]
    assert _reformat_to_latlong((np.nan, '1'), use_list=True) == [np.nan, 1]

    # This is how csv and parquet will deserialize
    assert _reformat_to_latlong('[1, nan]', use_list=True) == [1, np.nan]
    assert _reformat_to_latlong('[1, NaN]', use_list=True) == [1, np.nan]


def test_to_latlong_float():
    assert _to_latlong_float(4) == 4.0
    assert _to_latlong_float('2.2') == 2.2

    assert _to_latlong_float(None) is np.nan
    assert _to_latlong_float(np.nan) is np.nan
    assert _to_latlong_float(pd.NA) is np.nan

    error = re.escape('Latitude and Longitude values must be in decimal degrees. The latitude or longitude represented by [1, 2, 3] cannot be converted to a float.')
    with pytest.raises(ValueError, match=error):
        _to_latlong_float([1, 2, 3])


def test_is_null_latlong():
    assert _is_null_latlong(None)
    assert _is_null_latlong(np.nan)
    assert _is_null_latlong(pd.NA)
    assert _is_null_latlong('None')
    assert _is_null_latlong('nan')
    assert _is_null_latlong('NaN')

    assert not _is_null_latlong([None, 1, 3])
    assert not _is_null_latlong('none')
    assert not _is_null_latlong(0)
    assert not _is_null_latlong(False)


def test_is_valid_latlong_value():
    values = [
        (1.0, 2.0),
        np.nan,
        [1.0, 2.0],
        (np.nan, np.nan),
        ('a', 2.0),
        (1.0, 2.0, 3.0),
        None
    ]

    expected_values = [
        True,
        True,
        False,
        False,
        False,
        False,
        False
    ]

    for index, value in enumerate(values):
        assert _is_valid_latlong_value(value) is expected_values[index]


def test_is_valid_latlong_value_koalas():
    values = [
        (1.0, 2.0),
        np.nan,
        [1.0, 2.0],
        (np.nan, np.nan),
        ('a', 2.0),
        (1.0, 2.0, 3.0),
        None
    ]

    expected_values = [
        False,
        True,
        True,
        False,
        False,
        False,
        False
    ]

    for index, value in enumerate(values):
        assert _is_valid_latlong_value(value, bracket_type=list) is expected_values[index]


def test_is_valid_latlong_series():
    valid_series = pd.Series([(1.0, 2.0), (3.0, 4.0)])
    invalid_series = pd.Series([(1.0, 2.0), (3.0, '4.0')])

    assert _is_valid_latlong_series(valid_series) is True
    assert _is_valid_latlong_series(invalid_series) is False


def test_get_valid_mi_types():
    valid_types = get_valid_mi_types()
    expected_types = [
        Age,
        AgeNullable,
        Boolean,
        BooleanNullable,
        Categorical,
        CountryCode,
        Datetime,
        Double,
        Integer,
        IntegerNullable,
        Ordinal,
        PostalCode,
        SubRegionCode,
    ]

    assert valid_types == expected_types


def test_get_column_logical_type(sample_series):
    assert _get_column_logical_type(sample_series, None, 'col_name') == Categorical

    assert _get_column_logical_type(sample_series, Datetime, 'col_name') == Datetime


def test_parse_logical_type():
    assert _parse_logical_type('Datetime', 'col_name') == Datetime
    assert _parse_logical_type(Datetime, 'col_name') == Datetime

    ymd_format = Datetime(datetime_format='%Y-%m-%d')
    assert _parse_logical_type(ymd_format, 'col_name') == ymd_format


def test_parse_logical_type_errors():
    error = 'Must use an Ordinal instance with order values defined'
    with pytest.raises(TypeError, match=error):
        _parse_logical_type('Ordinal', 'col_name')

    with pytest.raises(TypeError, match=error):
        _parse_logical_type(Ordinal, 'col_name')

    error = "Invalid logical type specified for 'col_name'"
    with pytest.raises(TypeError, match=error):
        _parse_logical_type(int, 'col_name')


def test_concat_cols_ww_dfs(sample_df):
    sample_df.ww.init(logical_types={'full_name': 'Categorical'}, semantic_tags={'age': 'test_tag'},
                      table_metadata={'created_by': 'user0'})
    df1 = sample_df.ww[['id', 'full_name', 'email']]
    df2 = sample_df.ww[['phone_number', 'age', 'signup_date', 'is_registered']]

    combined_df = concat([df1, df2])
    assert combined_df.ww == sample_df.ww
    assert combined_df.ww.logical_types['full_name'] == Categorical
    assert 'test_tag' in combined_df.ww.semantic_tags['age']
    assert combined_df.ww.metadata == {'created_by': 'user0'}

    pandas_combined_df = pd.concat([to_pandas(df1), to_pandas(df2)], axis=1, join='outer', ignore_index=False)
    assert to_pandas(combined_df).equals(pandas_combined_df)


def test_concat_cols_uninit_dfs(sample_df):
    df1 = sample_df[['id', 'full_name', 'email']]
    df2 = sample_df[['phone_number', 'age', 'signup_date', 'is_registered']]

    combined_df = concat([df1, df2])
    assert df1.ww.schema is None
    assert df2.ww.schema is None
    assert combined_df.ww.schema is not None

    # Though the input dataframes don't have Woodwork initalized,
    # the concatenated DataFrame has Woodwork initialized for all the columns
    assert combined_df.ww.logical_types['id'] == Integer

    df1.ww.init()
    df2.ww.init()
    pandas_combined_df = pd.concat([to_pandas(df1), to_pandas(df2)], axis=1, join='outer', ignore_index=False)
    assert to_pandas(combined_df).equals(pandas_combined_df)


def test_concat_cols_combo_dfs(sample_df):
    df1 = sample_df[['id', 'full_name', 'email']]
    sample_df.ww.init()
    df2 = sample_df.ww[['phone_number', 'age', 'signup_date', 'is_registered']]

    combined_df = concat([df1, df2])
    assert df1.ww.schema is None
    assert combined_df.ww.logical_types['age'] == IntegerNullable

    df1.ww.init()
    df2.ww.init()
    pandas_combined_df = pd.concat([to_pandas(df1), to_pandas(df2)], axis=1, join='outer', ignore_index=False)
    assert to_pandas(combined_df).equals(pandas_combined_df)


def test_concat_cols_with_series(sample_df):
    df = sample_df[['id', 'full_name', 'email', 'phone_number', 'age']]
    s1 = sample_df['signup_date']
    sample_df.ww.init()
    s2 = sample_df['is_registered']

    combined_df = concat([df, s1, s2])

    df.ww.init()
    s1.ww.init()
    s2.ww.init()
    pandas_combined_df = pd.concat([to_pandas(df), to_pandas(s1), to_pandas(s2)], axis=1, join='outer', ignore_index=False)
    assert to_pandas(combined_df).equals(pandas_combined_df)


def test_concat_rows(sample_df):
    sample_df.ww.init()
    original_schema = sample_df.ww.schema
    copy_df = sample_df.ww.copy()

    combined_df = concat([sample_df, copy_df], axis=0)
    assert len(combined_df) == 8

    assert combined_df.ww.schema == original_schema


def test_concat_cols_with_conflicting_ww_indexes(sample_df):
    df1 = sample_df[['id', 'phone_number', 'email']]
    df1.ww.init(index='id')
    df2 = sample_df[['full_name', 'age', 'signup_date', 'is_registered']]
    df2.ww.init(index='full_name')

    error = 'Cannot concat dataframes with different Woodwork index columns.'
    with pytest.raises(IndexError, match=error):
        concat([df1, df2])

    df1 = sample_df[['id', 'phone_number', 'email']]
    df1.ww.init(time_index='id')
    df2 = sample_df[['full_name', 'age', 'signup_date', 'is_registered']]
    df2.ww.init(time_index='signup_date')

    error = 'Cannot concat dataframes with different Woodwork time index columns.'
    with pytest.raises(IndexError, match=error):
        concat([df1, df2])


def test_concat_cols_with_ww_indexes(sample_df):
    df1 = sample_df[['id', 'phone_number', 'email']]
    df1.ww.init(index='id')
    df2 = sample_df[['full_name', 'age', 'signup_date', 'is_registered']]
    df2.ww.init(time_index='signup_date')

    combined_df = concat([df1, df2])
    assert combined_df.ww.index == 'id'
    assert combined_df.ww.time_index == 'signup_date'


def test_concat_cols_with_duplicate_ww_indexes(sample_df):
    df1 = sample_df[['id', 'phone_number', 'signup_date', 'email']]
    df1.ww.init(index='id', time_index='signup_date')
    df2 = sample_df[['full_name', 'age', 'id', 'signup_date', 'is_registered']]
    df2.ww.init(index='id', time_index='signup_date')

    combined_df = concat([df1, df2])
    assert combined_df.ww.index == 'id'
    assert combined_df.ww.time_index == 'signup_date'
    assert len(set(combined_df.columns)) == len(set(sample_df.columns))


def test_concat_rows_with_ww_indexes(sample_df):
    sample_df.ww.init(index='id')
    original_schema = sample_df.ww.schema
    copy_df = sample_df.ww.copy()

    if isinstance(sample_df, pd.DataFrame):
        # Only pandas checks index uniqueness
        error = 'Index column must be unique'
        with pytest.raises(IndexError, match=error):
            concat([sample_df, copy_df], axis=0)

        # Without schema validation this does not get caught
        combined_df = concat([sample_df, copy_df], axis=0, validate_schema=False)
        assert combined_df.ww.index == 'id'

    sample_df.ww.init(time_index='signup_date')
    original_schema = sample_df.ww.schema
    copy_df = sample_df.ww.copy()

    combined_df = concat([sample_df, copy_df], axis=0)
    assert len(combined_df) == 8

    assert combined_df.ww.time_index == 'signup_date'
    assert combined_df.ww.schema == original_schema


# def test_concat_cols_with_different_underlying_index_dtypes(sample_df):
#     # --> this is very different for the diff dataframe types unless we catch the error ourselves
#     df1 = sample_df[['id', 'signup_date']]
#     df1.ww.init()
#     df2 = sample_df[['full_name', 'age']].set_index('full_name', drop=False)
#     df2.index.name = None
#     df2.ww.init()

#     assert df1.index.dtype == 'int64'
#     assert str(df2.index.dtype) == 'object'

#     error = 'Index column must be unique'
#     with pytest.raises(IndexError, match=error):
#         # --> might be better to have a different, more specialized error here??
#         concat([df1, df2])


def test_concat_table_names(sample_df):
    df1 = sample_df[['id', 'signup_date']]
    df1.ww.init(name=0)
    df2 = sample_df[['full_name', 'age']]
    df2.ww.init(name='1')
    df3 = sample_df[['phone_number', 'is_registered']]
    df3.ww.init(name='2')

    combined_df = concat([df1, df2, df3])
    assert combined_df.ww.name == '0_1_2'

    combined_df = concat([df3, df2, df1])
    assert combined_df.ww.name == '2_1_0'


def test_concat_cols_different_use_standard_tags(sample_df):
    sample_df.ww.init(use_standard_tags={'age': False, 'full_name': False, 'id': True},
                      logical_types={'full_name': 'Categorical'})

    assert sample_df.ww.semantic_tags['id'] == {'numeric'}
    assert sample_df.ww.semantic_tags['full_name'] == set()
    assert sample_df.ww.semantic_tags['age'] == set()

    df1 = sample_df.ww[['id', 'full_name', 'email']]
    df2 = sample_df.ww[['phone_number', 'age', 'signup_date', 'is_registered']]

    combined_df = concat([df1, df2])

    assert combined_df.ww.semantic_tags['id'] == {'numeric'}
    assert combined_df.ww.semantic_tags['full_name'] == set()
    assert combined_df.ww.semantic_tags['age'] == set()


def test_concat_rows_different_schema(sample_df):
    df1 = sample_df.copy()
    df1.ww.init(logical_types={'age': 'AgeNullable'})
    df2 = sample_df.copy()
    df2.ww.init(logical_types={'age': 'IntegerNullable'})
    df3 = sample_df.copy()
    df3.ww.init(logical_types={'age': 'NaturalLanguage'})

    assert df2.ww.schema != df1.ww.schema

    combined_df_1 = concat([df1, df2], axis=0)
    combined_df_1.ww.logical_types['age'] == AgeNullable

    combined_df_2 = concat([df2, df1], axis=0)
    combined_df_2.ww.logical_types['age'] == IntegerNullable

    combined_df_3 = concat([df3, df2], axis=0)
    combined_df_3.ww.logical_types['age'] == NaturalLanguage
    assert str(combined_df_3['age'].dtype) == 'string'


def test_concat_rows_different_columns(sample_df):
    sample_df.ww.init(time_index='signup_date')
    df1 = sample_df.ww[['id', 'signup_date', 'full_name', 'email']]
    df2 = sample_df.ww[['id', 'signup_date', 'phone_number', 'age', 'is_registered']]
    # should work if dtypes dont change

    combined_df = concat([df1, df2], axis=0)
    assert combined_df.ww.schema == sample_df.ww.schema
    assert to_pandas(combined_df).equals(pd.concat([to_pandas(df1), to_pandas(df2)]))


def test_concat_combine_metadatas(sample_df):
    df1 = sample_df[['id', 'full_name', 'email']]
    df1.ww.init(table_metadata={'created_by': 'user0', 'test_key': 'test_val1'}, column_metadata={'id': {'interesting_values': [1, 2]}})
    df2 = sample_df[['phone_number', 'age', 'signup_date', 'is_registered']]
    df2.ww.init(table_metadata={'table_type': 'single', 'test_key': 'test_val2'}, column_metadata={'age': {'interesting_values': [33]}})

    combined_df_1 = concat([df1, df2])

    # The first table sets the metadata when there's overlap
    assert combined_df_1.ww.metadata == {'created_by': 'user0',
                                         'table_type': 'single',
                                         'test_key': 'test_val1'}
    assert combined_df_1.ww.columns['id'].metadata == {'interesting_values': [1, 2]}
    assert combined_df_1.ww.columns['age'].metadata == {'interesting_values': [33]}

    combined_df_2 = concat([df2, df1])

    assert combined_df_2.ww.metadata == {'created_by': 'user0',
                                         'table_type': 'single',
                                         'test_key': 'test_val2'}
    assert combined_df_2.ww.columns['id'].metadata == {'interesting_values': [1, 2]}
    assert combined_df_2.ww.columns['age'].metadata == {'interesting_values': [33]}


@patch("woodwork.table_accessor._validate_accessor_params")
def test_concat_rows_validate_schema(mock_validate_accessor_params, sample_df):
    sample_df.ww.init(validate=False)
    copy_df = sample_df.ww.copy()

    assert not mock_validate_accessor_params.called

    concat([sample_df, copy_df], axis=0, validate_schema=False)

    assert not mock_validate_accessor_params.called

    concat([sample_df, copy_df], axis=0, validate_schema=True)

    assert mock_validate_accessor_params.called


def test_concat_cols_mismatched_index_adds_nans(sample_df):
    sample_df.ww.init(index='id')

    df1 = sample_df.ww.loc[[0, 1, 2], ['id', 'full_name']]
    df2 = sample_df.ww.loc[[1, 2, 3], ['id', 'email']]

    if isinstance(sample_df, pd.DataFrame):
        # Only pandas shows the dtype change because Dask and Koalas won't show until compute
        error = "Error converting datatype for id from type float64 to type int64. Please confirm the underlying data is consistent with logical type Integer."
        with pytest.raises(ww.exceptions.TypeConversionError, match=error):
            concat([df1, df2])

    # If the dtype can handle nans, it won't change
    sample_df.ww.init(index='id', logical_types={'id': 'IntegerNullable'})

    df1 = sample_df.ww.loc[[0, 1, 2], ['id', 'full_name']]
    df2 = sample_df.ww.loc[[1, 2, 3], ['id', 'email']]

    combined_df = concat([df1, df2])
    assert len(combined_df) == 4


def test_concat_cols_duplicates(sample_df):
    df1 = sample_df[['id', 'full_name', 'age']]
    df2 = sample_df[['id', 'email', 'signup_date']]

    # different error for Koalas because it gets caught at concat
    if ks and isinstance(sample_df, ks.DataFrame):
        error_type = ValueError
        error = re.escape("Labels have to be unique; however, got duplicated labels ['id'].")
    else:
        error_type = IndexError
        error = 'Dataframe cannot contain duplicate columns names'
    with pytest.raises(error_type, match=error):
        concat([df1, df2])


def test_concat_axis_params(sample_df):
    df1 = sample_df[['id', 'full_name', 'age']]
    df2 = sample_df[['email', 'signup_date']]
    df1.ww.init(logical_types={'id': 'IntegerNullable'})
    df2.ww.init()

    axis_0 = concat([df1, df2], axis=0)
    axis_index = concat([df1, df2], axis='index')
    assert axis_0.ww == axis_index.ww
    assert to_pandas(axis_0).equals(to_pandas(axis_index))

    axis_1 = concat([df1, df2], axis=1)
    axis_columns = concat([df1, df2], axis='columns')
    assert axis_1.ww == axis_columns.ww
    assert to_pandas(axis_1).equals(to_pandas(axis_columns))

    error = 'No axis named test'
    with pytest.raises(ValueError, match=error):
        concat([df1, df2], axis='test')


def test_concat_different_table_types(sample_df_pandas, sample_df_dask):
    sample_df_pandas.ww.init()
    sample_df_dask.ww.init()

    error = re.escape("cannot concatenate object of type '<class 'dask.dataframe.core.DataFrame'>'; only Series and DataFrame objs are valid")
    with pytest.raises(TypeError, match=error):
        concat([sample_df_dask, sample_df_pandas], axis=0)


def test_concat_cols_inner_join(sample_df):
    df1 = sample_df[['id', 'full_name', 'age']]
    df2 = sample_df[['email', 'signup_date']]


def test_concat_rows_inner_join(sample_df):
    # --> broken because losing columns breaks the i
    df1 = sample_df[['signup_date', 'email', 'age']]
    df1.ww.init(time_index='signup_date')

    df2 = sample_df[['id', 'email', 'signup_date', 'age']]
    df2.ww.init(index='id', time_index='signup_date')

    # If we lose a column in the inner join, we lose its typing information
    inner_combined = concat([df1, df2], join='inner', axis=0)
    assert set(inner_combined.columns) == {'email', 'age', 'signup_date'}
    assert inner_combined.ww.index is None
    assert inner_combined.ww.time_index == 'signup_date'


def test_concat_inner_join_no_overlap(sample_df):
    # --> test rows and cols
    pass


def test_concat_cols_table_order(sample_df):
    pass


def test_concat_rows_table_order(sample_df):
    pass


def test_concat_all_series(sample_df):
    # both rows and cols
    pass


def test_concat_empty_list():
    error = 'No objects to concatenate'
    with pytest.raises(ValueError, match=error):
        concat([])


def test_concat_with_none(sample_df):
    df1 = sample_df[['id', 'full_name', 'email']]
    df2 = sample_df[['phone_number', 'age', 'signup_date', 'is_registered']]

    error = 'Cannont include None in list of objects to concatenate'
    with pytest.raises(ValueError, match=error):
        combined_df_with_none = concat([df1, df2, None])
