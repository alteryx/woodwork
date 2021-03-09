import re

import numpy as np
import pandas as pd
import pytest

import woodwork as ww
from woodwork.exceptions import (
    ColumnNotPresentError,
    ParametersIgnoredWarning,
    TypeConversionError,
    TypingInfoMismatchWarning
)
from woodwork.logical_types import (
    URL,
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    EmailAddress,
    Filepath,
    FullName,
    Integer,
    IPAddress,
    LatLong,
    NaturalLanguage,
    Ordinal,
    PhoneNumber,
    SubRegionCode,
    ZIPCode
)
from woodwork.schema import Schema
from woodwork.table_accessor import (
    _check_index,
    _check_logical_types,
    _check_time_index,
    _check_unique_column_names,
    _get_invalid_schema_message
)
from woodwork.tests.testing_utils import to_pandas, validate_subset_schema
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_check_index_errors(sample_df):
    error_message = 'Specified index column `foo` not found in dataframe. To create a new index column, set make_index to True.'
    with pytest.raises(LookupError, match=error_message):
        _check_index(dataframe=sample_df, index='foo')

    if isinstance(sample_df, pd.DataFrame):
        # Does not check for index uniqueness with Dask
        error_message = 'Index column must be unique'
        with pytest.raises(LookupError, match=error_message):
            _check_index(sample_df, index='age')

    error_message = 'When setting make_index to True, the name specified for index cannot match an existing column name'
    with pytest.raises(IndexError, match=error_message):
        _check_index(sample_df, index='id', make_index=True)

    error_message = 'When setting make_index to True, the name for the new index must be specified in the index parameter'
    with pytest.raises(IndexError, match=error_message):
        _check_index(sample_df, index=None, make_index=True)


def test_check_logical_types_errors(sample_df):
    error_message = 'logical_types must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_logical_types(sample_df, logical_types='type')

    bad_logical_types_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("logical_types contains columns that are not present in dataframe: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_logical_types(sample_df, bad_logical_types_keys)


def test_check_time_index_errors(sample_df):
    error_message = 'Specified time index column `foo` not found in dataframe'
    with pytest.raises(LookupError, match=error_message):
        _check_time_index(dataframe=sample_df, time_index='foo')


def test_check_unique_column_names_errors(sample_df):
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.skip("Koalas enforces unique column names")
    duplicate_cols_df = sample_df.copy()
    if dd and isinstance(sample_df, dd.DataFrame):
        duplicate_cols_df = dd.concat([duplicate_cols_df, duplicate_cols_df['age']], axis=1)
    else:
        duplicate_cols_df.insert(0, 'age', [18, 21, 65, 43], allow_duplicates=True)
    with pytest.raises(IndexError, match='Dataframe cannot contain duplicate columns names'):
        _check_unique_column_names(duplicate_cols_df)


def test_accessor_init(sample_df):
    assert sample_df.ww.schema is None
    sample_df.ww.init()
    assert isinstance(sample_df.ww.schema, Schema)


def test_accessor_schema_property(sample_df):
    sample_df.ww.init()

    assert sample_df.ww._schema is not sample_df.ww.schema
    assert sample_df.ww._schema == sample_df.ww.schema


def test_accessor_physical_types_property(sample_df):
    sample_df.ww.init()

    assert isinstance(sample_df.ww.physical_types, dict)
    assert set(sample_df.ww.physical_types.keys()) == set(sample_df.columns)
    for k, v in sample_df.ww.physical_types.items():
        logical_type = sample_df.ww.columns[k]['logical_type']
        if ks and isinstance(sample_df, ks.DataFrame) and logical_type.backup_dtype is not None:
            assert v == logical_type.backup_dtype
        else:
            assert v == logical_type.primary_dtype


def test_accessor_separation_of_params(sample_df):
    # mix up order of acccessor and schema params
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_name', index='id', semantic_tags={'id': 'test_tag'}, time_index='signup_date')

    assert schema_df.ww.semantic_tags['id'] == {'index', 'test_tag'}
    assert schema_df.ww.index == 'id'
    assert schema_df.ww.time_index == 'signup_date'
    assert schema_df.ww.name == 'test_name'


def test_init_accessor_with_schema(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema', semantic_tags={'id': 'test_tag'}, index='id')
    schema = schema_df.ww._schema

    head_df = schema_df.head(2)
    assert head_df.ww.schema is None
    head_df.ww.init(schema=schema)

    assert head_df.ww._schema is schema
    assert head_df.ww.name == 'test_schema'
    assert head_df.ww.semantic_tags['id'] == {'index', 'test_tag'}

    iloc_df = schema_df.loc[[2, 3]]
    assert iloc_df.ww.schema is None
    iloc_df.ww.init(schema=schema)

    assert iloc_df.ww._schema is schema
    assert iloc_df.ww.name == 'test_schema'
    assert iloc_df.ww.semantic_tags['id'] == {'index', 'test_tag'}
    # Extra parameters do not take effect
    assert iloc_df.ww.logical_types['id'] == Integer


def test_init_accessor_with_schema_errors(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init()
    schema = schema_df.ww.schema

    iloc_df = schema_df.iloc[:, :-1]
    assert iloc_df.ww.schema is None

    error = 'Provided schema must be a Woodwork.Schema object.'
    with pytest.raises(TypeError, match=error):
        iloc_df.ww.init(schema=int)

    error = ("Woodwork typing information is not valid for this DataFrame: "
             "The following columns in the typing information were missing from the DataFrame: {'is_registered'}")
    with pytest.raises(ValueError, match=error):
        iloc_df.ww.init(schema=schema)


def test_accessor_with_schema_parameter_warning(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema', semantic_tags={'id': 'test_tag'}, index='id')
    schema = schema_df.ww.schema

    head_df = schema_df.head(2)

    warning = "A schema was provided and the following parameters were ignored: index, make_index, " \
              "time_index, logical_types, already_sorted, semantic_tags"
    with pytest.warns(ParametersIgnoredWarning, match=warning):
        head_df.ww.init(index='ignored_id', time_index="ignored_time_index", logical_types={'ignored': 'ltypes'},
                        make_index=True, already_sorted=True, semantic_tags={'ignored_id': 'ignored_test_tag'},
                        schema=schema)

    assert head_df.ww.name == 'test_schema'
    assert head_df.ww.semantic_tags['id'] == {'index', 'test_tag'}


def test_accessor_getattr(sample_df):
    schema_df = sample_df.copy()

    # We can access attributes on the Accessor class before the Schema is initialized
    assert schema_df.ww.schema is None

    error = re.escape("Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init")
    with pytest.raises(AttributeError, match=error):
        schema_df.ww.index

    schema_df.ww.init()

    assert schema_df.ww.name is None
    assert schema_df.ww.index is None
    assert schema_df.ww.time_index is None

    assert set(schema_df.ww.columns.keys()) == set(sample_df.columns)

    error = re.escape("Woodwork has no attribute 'not_present'")
    with pytest.raises(AttributeError, match=error):
        sample_df.ww.init()
        sample_df.ww.not_present


def test_getitem(sample_df):
    df = sample_df
    df.ww.init(
        time_index='signup_date',
        index='id', name='dt_name',
        logical_types={'age': 'Double'},
        semantic_tags={'age': {'custom_tag'}},
    )
    assert list(df.columns) == list(df.ww.schema.columns)

    subset = ['id', 'signup_date']
    df_subset = df.ww[subset]
    pd.testing.assert_frame_equal(to_pandas(df[subset]), to_pandas(df_subset))
    assert subset == list(df_subset.ww._schema.columns)
    assert df_subset.ww.index == 'id'
    assert df_subset.ww.time_index == 'signup_date'

    subset = ['age', 'email']
    df_subset = df.ww[subset]
    pd.testing.assert_frame_equal(to_pandas(df[subset]), to_pandas(df_subset))
    assert subset == list(df_subset.ww._schema.columns)
    assert df_subset.ww.index is None
    assert df_subset.ww.time_index is None
    assert df_subset.ww.logical_types['age'] == Double
    assert df_subset.ww.semantic_tags['age'] == {'custom_tag', 'numeric'}

    subset = df.ww[[]]
    assert len(subset.ww.columns) == 0
    assert subset.ww.index is None
    assert subset.ww.time_index is None

    series = df.ww['age']
    pd.testing.assert_series_equal(to_pandas(series), to_pandas(df['age']))
    assert series.ww.logical_type == Double
    assert series.ww.semantic_tags == {'custom_tag', 'numeric'}

    series = df.ww['id']
    pd.testing.assert_series_equal(to_pandas(series), to_pandas(df['id']))
    assert series.ww.logical_type == Integer
    assert series.ww.semantic_tags == {'numeric'}


def test_getitem_invalid_input(sample_df):
    df = sample_df
    df.ww.init()

    error_msg = r"Column\(s\) '\[1, 2\]' not found in DataFrame"
    with pytest.raises(ColumnNotPresentError, match=error_msg):
        df.ww[['email', 2, 1]]

    error_msg = "Column with name 'invalid_column' not found in DataFrame"
    with pytest.raises(ColumnNotPresentError, match=error_msg):
        df.ww['invalid_column']


def test_accessor_equality_with_schema(sample_df, sample_column_names, sample_inferred_logical_types):
    comparison_schema = Schema(sample_column_names, sample_inferred_logical_types)

    schema_df = sample_df.copy()
    schema_df.ww.init()

    # eq not implemented on Accessor class, so Schema's eq is called
    assert schema_df.ww.__eq__(comparison_schema)

    logical_types = {
        'id': Double,
        'full_name': FullName
    }
    semantic_tags = {
        'email': 'test_tag',
    }
    comparison_schema = Schema(sample_column_names,
                               logical_types={**sample_inferred_logical_types, **logical_types},
                               index='id',
                               time_index='signup_date',
                               semantic_tags=semantic_tags)
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types=logical_types,
                      index='id',
                      time_index='signup_date',
                      semantic_tags=semantic_tags,
                      already_sorted=True)
    assert schema_df.ww == comparison_schema


def test_accessor_init_with_valid_string_time_index(time_index_df):
    time_index_df.ww.init(name='schema',
                          index='id',
                          time_index='times')

    assert time_index_df.ww.name == 'schema'
    assert time_index_df.ww.index == 'id'
    assert time_index_df.ww.time_index == 'times'
    assert time_index_df.ww.columns[time_index_df.ww.time_index]['logical_type'] == Datetime


def test_accessor_init_with_numeric_datetime_time_index(time_index_df):
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints', logical_types={'ints': Datetime})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        time_index_df.ww.init(name='schema', time_index='strs', logical_types={'strs': Datetime})

    assert schema_df.ww.time_index == 'ints'
    assert schema_df['ints'].dtype == 'datetime64[ns]'


def test_accessor_with_numeric_time_index(time_index_df):
    # Set a numeric time index on init
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints')
    date_col = schema_df.ww.columns['ints']
    assert schema_df.ww.time_index == 'ints'
    assert date_col['logical_type'] == Integer
    assert date_col['semantic_tags'] == {'time_index', 'numeric'}

    # Specify logical type for time index on init
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints', logical_types={'ints': 'Double'})
    date_col = schema_df.ww.columns['ints']
    assert schema_df.ww.time_index == 'ints'
    assert date_col['logical_type'] == Double
    assert date_col['semantic_tags'] == {'time_index', 'numeric'}

    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='strs', logical_types={'strs': 'Double'})
    date_col = schema_df.ww.columns['strs']
    assert schema_df.ww.time_index == 'strs'
    assert date_col['logical_type'] == Double
    assert date_col['semantic_tags'] == {'time_index', 'numeric'}

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        time_index_df.ww.init(time_index='ints', logical_types={'ints': 'Categorical'})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        time_index_df.ww.init(time_index='letters', logical_types={'strs': 'Integer'})

    # Set numeric time index after init
    schema_df = time_index_df.copy()
    schema_df.ww.init(logical_types={'ints': 'Double'})
    assert schema_df.ww.time_index is None

    schema_df.ww.set_time_index('ints')
    date_col = schema_df.ww.columns['ints']
    assert schema_df.ww.time_index == 'ints'
    assert date_col['logical_type'] == Double
    assert date_col['semantic_tags'] == {'numeric', 'time_index'}


def test_accessor_init_with_invalid_string_time_index(sample_df):
    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        sample_df.ww.init(name='schema', time_index='full_name')


def test_accessor_init_with_string_logical_types(sample_df):
    logical_types = {
        'full_name': 'natural_language',
        'age': 'Double'
    }
    schema_df = sample_df.copy()
    schema_df.ww.init(name='schema',
                      logical_types=logical_types)
    assert schema_df.ww.columns['full_name']['logical_type'] == NaturalLanguage
    assert schema_df.ww.columns['age']['logical_type'] == Double

    logical_types = {
        'full_name': 'NaturalLanguage',
        'age': 'Integer',
        'signup_date': 'Datetime'
    }
    schema_df = sample_df.copy()
    schema_df.ww.init(name='schema',
                      logical_types=logical_types,
                      time_index='signup_date'
                      )
    assert schema_df.ww.columns['full_name']['logical_type'] == NaturalLanguage
    assert schema_df.ww.columns['age']['logical_type'] == Integer
    assert schema_df.ww.time_index == 'signup_date'


def test_int_dtype_inference_on_init():
    df = pd.DataFrame({
        'ints_no_nans': pd.Series([1, 2]),
        'ints_nan': pd.Series([1, np.nan]),
        'ints_NA': pd.Series([1, pd.NA]),
        'ints_NA_specified': pd.Series([1, pd.NA], dtype='Int64')})
    df.ww.init()

    assert df['ints_no_nans'].dtype == 'Int64'
    assert df['ints_nan'].dtype == 'float64'
    assert df['ints_NA'].dtype == 'category'
    assert df['ints_NA_specified'].dtype == 'Int64'


def test_bool_dtype_inference_on_init():
    df = pd.DataFrame({
        'bools_no_nans': pd.Series([True, False]),
        'bool_nan': pd.Series([True, np.nan]),
        'bool_NA': pd.Series([True, pd.NA]),
        'bool_NA_specified': pd.Series([True, pd.NA], dtype="boolean")})
    df.ww.init()

    assert df['bools_no_nans'].dtype == 'boolean'
    assert df['bool_nan'].dtype == 'category'
    assert df['bool_NA'].dtype == 'category'
    assert df['bool_NA_specified'].dtype == 'boolean'


def test_str_dtype_inference_on_init():
    df = pd.DataFrame({
        'str_no_nans': pd.Series(['a', 'b']),
        'str_nan': pd.Series(['a', np.nan]),
        'str_NA': pd.Series(['a', pd.NA]),
        'str_NA_specified': pd.Series([1, pd.NA], dtype="string"),
        'long_str_NA_specified': pd.Series(['this is a very long sentence inferred as a string', pd.NA], dtype="string"),
        'long_str_NA': pd.Series(['this is a very long sentence inferred as a string', pd.NA])
    })
    df.ww.init()

    assert df['str_no_nans'].dtype == 'category'
    assert df['str_nan'].dtype == 'category'
    assert df['str_NA'].dtype == 'category'
    assert df['str_NA_specified'].dtype == 'category'
    assert df['long_str_NA_specified'].dtype == 'string'
    assert df['long_str_NA'].dtype == 'string'


def test_float_dtype_inference_on_init():
    df = pd.DataFrame({
        'floats_no_nans': pd.Series([1.1, 2.2]),
        'floats_nan': pd.Series([1.1, np.nan]),
        'floats_NA': pd.Series([1.1, pd.NA]),
        'floats_nan_specified': pd.Series([1.1, np.nan], dtype='float')})
    df.ww.init()

    assert df['floats_no_nans'].dtype == 'float64'
    assert df['floats_nan'].dtype == 'float64'
    assert df['floats_NA'].dtype == 'category'
    assert df['floats_nan_specified'].dtype == 'float64'


def test_datetime_dtype_inference_on_init():
    df = pd.DataFrame({
        'date_no_nans': pd.Series([pd.to_datetime('2020-09-01')] * 2),
        'date_nan': pd.Series([pd.to_datetime('2020-09-01'), np.nan]),
        'date_NA': pd.Series([pd.to_datetime('2020-09-01'), pd.NA]),
        'date_NaT': pd.Series([pd.to_datetime('2020-09-01'), pd.NaT]),
        'date_NA_specified': pd.Series([pd.to_datetime('2020-09-01'), pd.NA], dtype='datetime64[ns]')})
    df.ww.init()

    assert df['date_no_nans'].dtype == 'datetime64[ns]'
    assert df['date_nan'].dtype == 'datetime64[ns]'
    assert df['date_NA'].dtype == 'datetime64[ns]'
    assert df['date_NaT'].dtype == 'datetime64[ns]'
    assert df['date_NA_specified'].dtype == 'datetime64[ns]'


def test_timedelta_dtype_inference_on_init():
    df = pd.DataFrame({
        'delta_no_nans': (pd.Series([pd.to_datetime('2020-09-01')] * 2) - pd.to_datetime('2020-07-01')),
        'delta_nan': (pd.Series([pd.to_datetime('2020-09-01'), np.nan]) - pd.to_datetime('2020-07-01')),
        'delta_NaT': (pd.Series([pd.to_datetime('2020-09-01'), pd.NaT]) - pd.to_datetime('2020-07-01')),
        'delta_NA_specified': (pd.Series([pd.to_datetime('2020-09-01'), pd.NA], dtype='datetime64[ns]') - pd.to_datetime('2020-07-01')),
    })
    df.ww.init()

    assert df['delta_no_nans'].dtype == 'timedelta64[ns]'
    assert df['delta_nan'].dtype == 'timedelta64[ns]'
    assert df['delta_NaT'].dtype == 'timedelta64[ns]'
    assert df['delta_NA_specified'].dtype == 'timedelta64[ns]'


def test_sets_category_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['a', 'b', 'c'], name=column_name),
        pd.Series(['a', None, 'c'], name=column_name),
        pd.Series(['a', np.nan, 'c'], name=column_name),
        pd.Series(['a', pd.NA, 'c'], name=column_name),
        pd.Series(['a', pd.NaT, 'c'], name=column_name),
    ]

    logical_types = [
        Categorical,
        CountryCode,
        Ordinal(order=['a', 'b', 'c']),
        SubRegionCode,
        ZIPCode,
    ]

    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            df = pd.DataFrame(series)
            df.ww.init(logical_types=ltypes)
            assert df.ww.columns[column_name]['logical_type'] == logical_type
            assert df[column_name].dtype == logical_type.primary_dtype


def test_sets_object_dtype_on_init(latlong_df):
    for column_name in latlong_df.columns:
        ltypes = {
            column_name: LatLong,
        }
        df = latlong_df.loc[:, [column_name]]
        df.ww.init(logical_types=ltypes)
        assert df.ww.columns[column_name]['logical_type'] == LatLong
        assert df[column_name].dtype == LatLong.primary_dtype
        df_pandas = to_pandas(df[column_name])
        expected_val = (3, 4)
        if ks and isinstance(latlong_df, ks.DataFrame):
            expected_val = [3, 4]
        assert df_pandas.iloc[-1] == expected_val


def test_sets_string_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['a', 'b', 'c'], name=column_name),
        pd.Series(['a', None, 'c'], name=column_name),
        pd.Series(['a', np.nan, 'c'], name=column_name),
        pd.Series(['a', pd.NA, 'c'], name=column_name),
    ]

    logical_types = [
        Filepath,
        FullName,
        IPAddress,
        NaturalLanguage,
        PhoneNumber,
        URL,
    ]

    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            df = pd.DataFrame(series)
            df.ww.init(logical_types=ltypes)
            assert df.ww.columns[column_name]['logical_type'] == logical_type
            assert df[column_name].dtype == logical_type.primary_dtype


def test_sets_boolean_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([True, False, True], name=column_name),
        pd.Series([True, None, True], name=column_name),
        pd.Series([True, np.nan, True], name=column_name),
        pd.Series([True, pd.NA, True], name=column_name),
    ]

    logical_type = Boolean
    for series in series_list:
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        df = pd.DataFrame(series)
        df.ww.init(logical_types=ltypes)
        assert df.ww.columns[column_name]['logical_type'] == logical_type
        assert df[column_name].dtype == logical_type.primary_dtype


def test_sets_int64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([1, 2, 3], name=column_name),
        pd.Series([1, None, 3], name=column_name),
        pd.Series([1, np.nan, 3], name=column_name),
        pd.Series([1, pd.NA, 3], name=column_name),
    ]

    logical_types = [Integer]
    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            df = pd.DataFrame(series)
            df.ww.init(logical_types=ltypes)
            assert df.ww.columns[column_name]['logical_type'] == logical_type
            assert df[column_name].dtype == logical_type.primary_dtype


def test_sets_float64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([1.1, 2, 3], name=column_name),
        pd.Series([1.1, None, 3], name=column_name),
        pd.Series([1.1, np.nan, 3], name=column_name),
    ]

    logical_type = Double
    for series in series_list:
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        df = pd.DataFrame(series)
        df.ww.init(logical_types=ltypes)
        assert df.ww.columns[column_name]['logical_type'] == logical_type
        assert df[column_name].dtype == logical_type.primary_dtype


def test_sets_datetime64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', None, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', np.nan, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', pd.NA, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', pd.NaT, '2020-01-03'], name=column_name),
    ]

    logical_type = Datetime
    for series in series_list:
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        df = pd.DataFrame(series)
        df.ww.init(logical_types=ltypes)
        assert df.ww.columns[column_name]['logical_type'] == logical_type
        assert df[column_name].dtype == logical_type.primary_dtype

    # Cannot cast a column with pd.NA to Double
    series = pd.Series([1.1, pd.NA, 3], name=column_name)
    ltypes = {
        column_name: Double,
    }
    err_msg = 'Error converting datatype for test_series from type object to type ' \
        'float64. Please confirm the underlying data is consistent with logical type Double.'
    with pytest.raises(TypeConversionError, match=err_msg):
        pd.DataFrame(series).ww.init(logical_types=ltypes)

    # Cannot cast Datetime to Double
    df = pd.DataFrame({column_name: ['2020-01-01', '2020-01-02', '2020-01-03']})
    df.ww.init(logical_types={column_name: Datetime})

    err_msg = 'Error converting datatype for test_series from type datetime64[ns] to type ' \
        'float64. Please confirm the underlying data is consistent with logical type Double.'
    with pytest.raises(TypeConversionError, match=re.escape(err_msg)):
        df.ww.set_types(logical_types={column_name: Double})

    # Cannot cast invalid strings to integers
    series = pd.Series(['1', 'two', '3'], name=column_name)
    ltypes = {
        column_name: Integer,
    }
    err_msg = 'Error converting datatype for test_series from type object to type ' \
        'Int64. Please confirm the underlying data is consistent with logical type Integer.'
    with pytest.raises(TypeConversionError, match=err_msg):
        pd.DataFrame(series).ww.init(logical_types=ltypes)


def test_make_index(sample_df):
    if ks and isinstance(sample_df, ks.DataFrame):
        error_msg = "Cannot make index on a Koalas DataFrame."
        with pytest.raises(TypeError, match=error_msg):
            sample_df.ww.init(index='new_index', make_index=True)
    else:
        schema_df = sample_df.copy()
        schema_df.ww.init(index='new_index', make_index=True)

        assert schema_df.ww.index == 'new_index'
        assert 'new_index' in schema_df.ww.columns
        assert 'new_index' in schema_df.ww.columns
        assert to_pandas(schema_df)['new_index'].unique
        assert to_pandas(schema_df['new_index']).is_monotonic
        assert 'index' in schema_df.ww.columns['new_index']['semantic_tags']


def test_underlying_index_set_no_index_on_init(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    input_index = pd.Int64Index([99, 88, 77, 66])

    schema_df = sample_df.copy()
    schema_df.index = input_index.copy()
    pd.testing.assert_index_equal(input_index, schema_df.index)

    schema_df.ww.init()
    assert schema_df.ww.index is None
    pd.testing.assert_index_equal(input_index, schema_df.index)

    sorted_df = schema_df.ww.sort_values('full_name')
    assert sorted_df.ww.index is None
    pd.testing.assert_index_equal(pd.Int64Index([88, 77, 99, 66]), sorted_df.index)


def test_underlying_index_set(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    # Sets underlying index at init
    schema_df = sample_df.copy()
    schema_df.ww.init(index='full_name')
    assert 'full_name' in schema_df.columns
    assert schema_df.index.name is None
    assert (schema_df.index == schema_df['full_name']).all()

    # Sets underlying index on update
    schema_df = sample_df.copy()
    schema_df.ww.init(index='id')
    schema_df.ww.set_index('full_name')
    assert schema_df.ww.index == 'full_name'
    assert 'full_name' in schema_df.columns
    assert (schema_df.index == schema_df['full_name']).all()
    assert schema_df.index.name is None

    # confirm removing Woodwork index doesn't change underlying index
    schema_df.ww.set_index(None)
    assert schema_df.ww.index is None
    assert (schema_df.index == schema_df['full_name']).all()

    # Sets underlying index for made index
    schema_df = sample_df.copy()
    schema_df.ww.init(index='made_index', make_index=True)
    assert 'made_index' in schema_df
    assert (schema_df.index == [0, 1, 2, 3]).all()
    assert schema_df.index.name is None


def test_underlying_index_reset(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    specified_index = pd.Index
    unspecified_index = pd.RangeIndex

    sample_df.ww.init()
    assert type(sample_df.index) == unspecified_index

    sample_df.ww.set_index('full_name')
    assert type(sample_df.index) == specified_index

    copied_df = sample_df.ww.copy()
    warning = ("Index mismatch between DataFrame and typing information")
    with pytest.warns(TypingInfoMismatchWarning, match=warning):
        copied_df.ww.reset_index(drop=True, inplace=True)
    assert copied_df.ww.schema is None
    assert type(copied_df.index) == unspecified_index

    sample_df.ww.set_index(None)
    assert type(sample_df.index) == specified_index

    # Use pandas operation to reset index
    reset_df = sample_df.ww.reset_index(drop=True, inplace=False)
    assert type(sample_df.index) == specified_index
    assert type(reset_df.index) == unspecified_index

    sample_df.ww.reset_index(drop=True, inplace=True)
    assert type(sample_df.index) == unspecified_index


def test_underlying_index_unchanged_after_updates(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    sample_df.ww.init(index='full_name')
    assert 'full_name' in sample_df
    assert sample_df.ww.index == 'full_name'
    assert (sample_df.index == sample_df['full_name']).all()

    copied_df = sample_df.ww.copy()

    dropped_df = copied_df.ww.drop('full_name')
    assert 'full_name' not in dropped_df
    assert dropped_df.ww.index is None
    assert (dropped_df.index == sample_df['full_name']).all()

    selected_df = copied_df.ww.select('Integer')
    assert 'full_name' not in dropped_df
    assert selected_df.ww.index is None
    assert (selected_df.index == sample_df['full_name']).all()

    iloc_df = copied_df.ww.iloc[:, 2:]
    assert 'full_name' not in iloc_df
    assert iloc_df.ww.index is None
    assert (iloc_df.index == sample_df['full_name']).all()

    loc_df = copied_df.ww.loc[:, ['id', 'email']]
    assert 'full_name' not in loc_df
    assert loc_df.ww.index is None
    assert (loc_df.index == sample_df['full_name']).all()

    subset_df = copied_df.ww[['id', 'email']]
    assert 'full_name' not in subset_df
    assert subset_df.ww.index is None
    assert (subset_df.index == sample_df['full_name']).all()

    reset_tags_df = sample_df.ww.copy()
    reset_tags_df.ww.reset_semantic_tags('full_name', retain_index_tags=False)
    assert reset_tags_df.ww.index is None
    assert (reset_tags_df.index == sample_df['full_name']).all()

    remove_tags_df = sample_df.ww.copy()
    remove_tags_df.ww.remove_semantic_tags({'full_name': 'index'})
    assert remove_tags_df.ww.index is None
    assert (remove_tags_df.index == sample_df['full_name']).all()

    set_types_df = sample_df.ww.copy()
    set_types_df.ww.set_types(semantic_tags={'full_name': 'new_tag'}, retain_index_tags=False)
    assert set_types_df.ww.index is None
    assert (set_types_df.index == sample_df['full_name']).all()

    popped_df = sample_df.ww.copy()
    popped_df.ww.pop('full_name')
    assert popped_df.ww.index is None
    assert (popped_df.index == sample_df['full_name']).all()


def test_accessor_already_sorted(sample_unsorted_df):
    if dd and isinstance(sample_unsorted_df, dd.DataFrame):
        pytest.xfail('Sorting dataframe is not supported with Dask input')
    if ks and isinstance(sample_unsorted_df, ks.DataFrame):
        pytest.xfail('Sorting dataframe is not supported with Koalas input')

    schema_df = sample_unsorted_df.copy()
    schema_df.ww.init(name='schema',
                      index='id',
                      time_index='signup_date')

    assert schema_df.ww.time_index == 'signup_date'
    assert schema_df.ww.columns[schema_df.ww.time_index]['logical_type'] == Datetime

    sorted_df = to_pandas(sample_unsorted_df).sort_values(['signup_date', 'id']).set_index('id', drop=False)
    sorted_df.index.name = None
    pd.testing.assert_frame_equal(sorted_df,
                                  to_pandas(schema_df), check_index_type=False, check_dtype=False)

    schema_df = sample_unsorted_df.copy()
    schema_df.ww.init(name='schema',
                      index='id',
                      time_index='signup_date',
                      already_sorted=True)

    assert schema_df.ww.time_index == 'signup_date'
    assert schema_df.ww.columns[schema_df.ww.time_index]['logical_type'] == Datetime

    unsorted_df = to_pandas(sample_unsorted_df.set_index('id', drop=False))
    unsorted_df.index.name = None
    pd.testing.assert_frame_equal(unsorted_df, to_pandas(schema_df), check_index_type=False, check_dtype=False)


def test_ordinal_with_order(sample_series):
    if (ks and isinstance(sample_series, ks.Series)) or (dd and isinstance(sample_series, dd.Series)):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not compatible')

    ordinal_with_order = Ordinal(order=['a', 'b', 'c'])
    schema_df = pd.DataFrame(sample_series)
    schema_df.ww.init(logical_types={'sample_series': ordinal_with_order})

    column_logical_type = schema_df.ww.logical_types['sample_series']
    assert isinstance(column_logical_type, Ordinal)
    assert column_logical_type.order == ['a', 'b', 'c']

    schema_df = pd.DataFrame(sample_series)
    schema_df.ww.init()

    schema_df.ww.set_types(logical_types={'sample_series': ordinal_with_order})
    logical_type = schema_df.ww.logical_types['sample_series']
    assert isinstance(logical_type, Ordinal)
    assert logical_type.order == ['a', 'b', 'c']


def test_ordinal_with_incomplete_ranking(sample_series):
    if (ks and isinstance(sample_series, ks.Series)) or (dd and isinstance(sample_series, dd.Series)):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not supported')

    ordinal_incomplete_order = Ordinal(order=['a', 'b'])
    error_msg = re.escape("Ordinal column sample_series contains values that are not "
                          "present in the order values provided: ['c']")

    schema_df = pd.DataFrame(sample_series)

    with pytest.raises(ValueError, match=error_msg):
        schema_df.ww.init(logical_types={'sample_series': ordinal_incomplete_order})

    schema_df.ww.init()
    with pytest.raises(ValueError, match=error_msg):
        schema_df.ww.set_types(logical_types={'sample_series': ordinal_incomplete_order})


def test_ordinal_with_nan_values():
    nan_df = pd.DataFrame(pd.Series(['a', 'b', np.nan, 'a'], name='nan_series'))
    ordinal_with_order = Ordinal(order=['a', 'b'])
    nan_df.ww.init(logical_types={'nan_series': ordinal_with_order})

    column_logical_type = nan_df.ww.logical_types['nan_series']
    assert isinstance(column_logical_type, Ordinal)
    assert column_logical_type.order == ['a', 'b']


def test_accessor_with_falsy_column_names(falsy_names_df):
    if dd and isinstance(falsy_names_df, dd.DataFrame):
        pytest.xfail('Dask DataFrames cannot handle integer column names')

    schema_df = falsy_names_df.copy()
    schema_df.ww.init(index=0, time_index='')
    assert schema_df.ww.index == 0
    assert schema_df.ww.time_index == ''

    schema_df.ww.set_time_index(None)
    assert schema_df.ww.time_index is None

    schema_df.ww.set_time_index('')
    assert schema_df.ww.time_index == ''

    schema_df.ww.pop('')
    assert '' not in schema_df
    assert '' not in schema_df.ww.columns
    assert schema_df.ww.time_index is None

    schema_df.ww.set_index(None)
    assert schema_df.ww.index is None

    # --> add back in if setitem is implemented
    # dt[''] = popped_col
    # assert dt[''].name == ''
    # assert dt['']._series.name == ''

    renamed_df = schema_df.ww.rename({0: 'col_with_name'})
    assert 0 not in renamed_df.columns
    assert 'col_with_name' in renamed_df.columns


def test_get_invalid_schema_message(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema', index='id', logical_types={'id': 'Double', 'full_name': 'FullName'})
    schema = schema_df.ww.schema

    assert _get_invalid_schema_message(schema_df, schema) is None
    assert (_get_invalid_schema_message(sample_df, schema) ==
            'dtype mismatch for column id between DataFrame dtype, int64, and Double dtype, float64')

    sampled_df = schema_df.sample(frac=0.3)
    assert _get_invalid_schema_message(sampled_df, schema) is None

    dropped_df = schema_df.drop('id', axis=1)
    assert (_get_invalid_schema_message(dropped_df, schema) ==
            "The following columns in the typing information were missing from the DataFrame: {'id'}")

    renamed_df = schema_df.rename(columns={'id': 'new_col'})
    assert (_get_invalid_schema_message(renamed_df, schema) ==
            "The following columns in the DataFrame were missing from the typing information: {'new_col'}")


def test_get_invalid_schema_message_dtype_mismatch(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types={'age': 'Categorical'})
    schema = schema_df.ww.schema

    incorrect_int_dtype_df = schema_df.ww.astype({'id': 'int64'})
    incorrect_bool_dtype_df = schema_df.ww.astype({'is_registered': 'bool'})
    incorrect_str_dtype_df = schema_df.ww.astype({'full_name': 'object'})  # wont work for koalas
    incorrect_categorical_dtype_df = schema_df.ww.astype({'age': 'string'})  # wont work for koalas

    assert (_get_invalid_schema_message(incorrect_int_dtype_df, schema) ==
            'dtype mismatch for column id between DataFrame dtype, int64, and Integer dtype, Int64')
    assert (_get_invalid_schema_message(incorrect_bool_dtype_df, schema) ==
            'dtype mismatch for column is_registered between DataFrame dtype, bool, and Boolean dtype, boolean')
    # Koalas backup dtypes make these checks not relevant
    if ks and not isinstance(sample_df, ks.DataFrame):
        assert (_get_invalid_schema_message(incorrect_str_dtype_df, schema) ==
                'dtype mismatch for column full_name between DataFrame dtype, object, and NaturalLanguage dtype, string')
        assert (_get_invalid_schema_message(incorrect_categorical_dtype_df, schema) ==
                'dtype mismatch for column age between DataFrame dtype, string, and Categorical dtype, category')


def test_get_invalid_schema_message_index_checks(sample_df):
    if not isinstance(sample_df, pd.DataFrame):
        pytest.xfail('Index validation not performed for Dask or Koalas DataFrames')

    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema', index='id', logical_types={'id': 'Double', 'full_name': 'FullName'})
    schema = schema_df.ww.schema

    different_underlying_index_df = schema_df.copy()
    different_underlying_index_df['id'] = pd.Series([9, 8, 7, 6], dtype='float64')
    assert (_get_invalid_schema_message(different_underlying_index_df, schema) ==
            "Index mismatch between DataFrame and typing information")

    not_unique_df = schema_df.replace({3: 1})
    not_unique_df.index = not_unique_df['id']
    not_unique_df.index.name = None
    assert _get_invalid_schema_message(not_unique_df, schema) == 'Index column is not unique'


def test_dataframe_methods_on_accessor(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema')

    copied_df = schema_df.ww.copy()

    assert schema_df is not copied_df
    assert schema_df.ww._schema is not copied_df.ww._schema
    assert copied_df.ww.schema == schema_df.ww.schema

    pd.testing.assert_frame_equal(to_pandas(schema_df), to_pandas(copied_df))

    ltype_dtype = 'Int64'
    new_dtype = 'string'

    warning = 'Operation performed by astype has invalidated the Woodwork typing information:\n '\
        f'dtype mismatch for column id between DataFrame dtype, {new_dtype}, and Integer dtype, {ltype_dtype}.\n '\
        'Please initialize Woodwork with DataFrame.ww.init'
    with pytest.warns(TypingInfoMismatchWarning, match=warning):
        new_df = schema_df.ww.astype({'id': new_dtype})
    assert new_df['id'].dtype == new_dtype
    assert new_df.ww.schema is None
    assert schema_df.ww.schema is not None


def test_dataframe_methods_on_accessor_new_schema_object(sample_df):
    sample_df.ww.init(index='id', semantic_tags={'email': 'new_tag'},
                      table_metadata={'contributors': ['user1', 'user2'],
                                      'created_on': '2/12/20'},
                      column_metadata={'id': {'important_keys': [1, 2, 3]}})

    copied_df = sample_df.ww.copy()

    assert sample_df.ww.schema == copied_df.ww.schema
    assert sample_df.ww._schema is not copied_df.ww._schema

    copied_df.ww.metadata['contributors'].append('user3')
    assert copied_df.ww.metadata == {'contributors': ['user1', 'user2', 'user3'],
                                     'created_on': '2/12/20'}

    assert sample_df.ww.metadata == {'contributors': ['user1', 'user2'],
                                     'created_on': '2/12/20'}

    copied_df.ww.reset_semantic_tags(retain_index_tags=False)
    assert copied_df.ww.index is None
    assert sample_df.ww.index == 'id'

    assert copied_df.ww.semantic_tags['email'] == set()
    assert sample_df.ww.semantic_tags['email'] == {'new_tag'}

    copied_df.ww.columns['id']['metadata']['important_keys'].append(4)
    assert copied_df.ww.columns['id']['metadata'] == {'important_keys': [1, 2, 3, 4]}
    assert sample_df.ww.columns['id']['metadata'] == {'important_keys': [1, 2, 3]}


def test_dataframe_methods_on_accessor_inplace(sample_df):
    # TODO: Try to find a supported inplace method for Dask, if one exists
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Dask does not support sort_values or rename inplace.')
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema')

    df_pre_sort = schema_df.copy()

    schema_df.ww.sort_values(['full_name'], inplace=True)
    assert schema_df.ww.name == 'test_schema'

    pd.testing.assert_frame_equal(to_pandas(schema_df), to_pandas(df_pre_sort.sort_values(['full_name'])))

    warning = "Operation performed by insert has invalidated the Woodwork typing information:\n "
    "The following columns in the DataFrame were missing from the typing information: {'new_name'}.\n "
    "Please initialize Woodwork with DataFrame.ww.init"
    with pytest.warns(TypingInfoMismatchWarning, match=warning):
        schema_df.ww.insert(loc=0, column="new_name", value=[1, 2, 3, 4])
    assert 'new_name' in schema_df.columns
    assert schema_df.ww.schema is None


def test_dataframe_methods_on_accessor_returning_series(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema')

    dtypes = schema_df.ww.dtypes

    assert schema_df.ww.name == 'test_schema'
    pd.testing.assert_series_equal(dtypes, schema_df.dtypes)
    all_series = schema_df.ww.all()
    assert schema_df.ww.name == 'test_schema'
    pd.testing.assert_series_equal(to_pandas(all_series), to_pandas(schema_df.all()))


def test_dataframe_methods_on_accessor_other_returns(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema')

    shape = schema_df.ww.shape

    assert schema_df.ww.name == 'test_schema'
    if dd and isinstance(sample_df, dd.DataFrame):
        shape = (shape[0].compute(), shape[1])
    assert shape == to_pandas(schema_df).shape
    assert schema_df.ww.name == 'test_schema'

    if dd and not isinstance(sample_df, dd.DataFrame):
        # keys() not supported with Dask
        pd.testing.assert_index_equal(schema_df.ww.keys(), schema_df.keys())


def test_dataframe_methods_on_accessor_to_pandas(sample_df):
    if isinstance(sample_df, pd.DataFrame):
        pytest.skip("No need to test converting pandas DataFrame to pandas")

    sample_df.ww.init(name='woodwork', index='id')

    if dd and isinstance(sample_df, dd.DataFrame):
        pd_df = sample_df.ww.compute()
        assert isinstance(pd_df, pd.DataFrame)
        assert pd_df.ww.index == 'id'
        assert pd_df.ww.name == 'woodwork'

    elif ks and isinstance(sample_df, ks.DataFrame):
        pd_df = sample_df.ww.to_pandas()
        assert isinstance(pd_df, pd.DataFrame)
        assert pd_df.ww.index == 'id'
        assert pd_df.ww.name == 'woodwork'


def test_get_subset_df_with_schema(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={'full_name': FullName,
                                     'email': EmailAddress,
                                     'phone_number': PhoneNumber,
                                     'age': Double,
                                     'signup_date': Datetime},
                      semantic_tags={'full_name': ['new_tag', 'tag2'],
                                     'age': 'numeric'}
                      )
    schema = schema_df.ww.schema

    empty_df = schema_df.ww._get_subset_df_with_schema([])
    assert len(empty_df.columns) == 0
    assert empty_df.ww.schema is not None
    pd.testing.assert_frame_equal(to_pandas(empty_df), to_pandas(schema_df[[]]))
    validate_subset_schema(empty_df.ww.schema, schema)

    just_index = schema_df.ww._get_subset_df_with_schema(['id'])
    assert just_index.ww.index == schema.index
    assert just_index.ww.time_index is None
    pd.testing.assert_frame_equal(to_pandas(just_index), to_pandas(schema_df[['id']]))
    validate_subset_schema(just_index.ww.schema, schema)

    just_time_index = schema_df.ww._get_subset_df_with_schema(['signup_date'])
    assert just_time_index.ww.time_index == schema.time_index
    assert just_time_index.ww.index is None
    pd.testing.assert_frame_equal(to_pandas(just_time_index), to_pandas(schema_df[['signup_date']]))
    validate_subset_schema(just_time_index.ww.schema, schema)

    transfer_schema = schema_df.ww._get_subset_df_with_schema(['phone_number'])
    assert transfer_schema.ww.index is None
    assert transfer_schema.ww.time_index is None
    pd.testing.assert_frame_equal(to_pandas(transfer_schema), to_pandas(schema_df[['phone_number']]))
    validate_subset_schema(transfer_schema.ww.schema, schema)


def test_get_subset_df_use_dataframe_order(sample_df):
    df = sample_df
    columns = list(df.columns)
    df.ww.init()

    reverse = list(reversed(columns))
    actual = df.ww._get_subset_df_with_schema(reverse, use_dataframe_order=True)
    assert list(actual.columns) == columns
    actual = df.ww._get_subset_df_with_schema(reverse, use_dataframe_order=False)
    assert list(actual.columns) == reverse


def test_select_ltypes_no_match_and_all(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types={'full_name': FullName,
                                     'email': EmailAddress,
                                     'phone_number': PhoneNumber,
                                     'age': Double,
                                     'signup_date': Datetime,
                                     })

    assert len(schema_df.ww.select(ZIPCode).columns) == 0
    assert len(schema_df.ww.select(['ZIPCode', PhoneNumber]).columns) == 1

    all_types = ww.type_system.registered_types
    df_all_types = schema_df.ww.select(all_types)

    pd.testing.assert_frame_equal(to_pandas(df_all_types), to_pandas(schema_df))
    assert df_all_types.ww.schema == schema_df.ww.schema


def test_select_ltypes_strings(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types={'full_name': FullName,
                                     'email': EmailAddress,
                                     'phone_number': PhoneNumber,
                                     'age': Double,
                                     'signup_date': Datetime,
                                     })

    df_multiple_ltypes = schema_df.ww.select(['FullName', 'email_address', 'double', 'Boolean', 'datetime'])
    assert len(df_multiple_ltypes.columns) == 5
    assert 'phone_number' not in df_multiple_ltypes.columns
    assert 'id' not in df_multiple_ltypes.columns

    df_single_ltype = schema_df.ww.select('full_name')
    assert set(df_single_ltype.columns) == {'full_name'}


def test_select_ltypes_objects(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types={'full_name': FullName,
                                     'email': EmailAddress,
                                     'phone_number': PhoneNumber,
                                     'age': Double,
                                     'signup_date': Datetime,
                                     })

    df_multiple_ltypes = schema_df.ww.select([FullName, EmailAddress, Double, Boolean, Datetime])
    assert len(df_multiple_ltypes.columns) == 5
    assert 'phone_number' not in df_multiple_ltypes.columns
    assert 'id' not in df_multiple_ltypes.columns

    df_single_ltype = schema_df.ww.select(FullName)
    assert len(df_single_ltype.columns) == 1


def test_select_ltypes_mixed(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types={'full_name': FullName,
                                     'email': EmailAddress,
                                     'phone_number': PhoneNumber,
                                     'age': Double,
                                     'signup_date': Datetime,
                                     })

    df_mixed_ltypes = schema_df.ww.select(['FullName', 'email_address', Double])
    assert len(df_mixed_ltypes.columns) == 3
    assert 'phone_number' not in df_mixed_ltypes.columns


def test_select_ltypes_table(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='testing',
                      index='id',
                      time_index='signup_date',
                      logical_types={'full_name': FullName,
                                     'email': EmailAddress,
                                     'phone_number': PhoneNumber,
                                     'age': Double,
                                     'signup_date': Datetime,
                                     },
                      semantic_tags={'full_name': ['new_tag', 'tag2'],
                                     'age': 'numeric',
                                     })

    df_no_indices = schema_df.ww.select('phone_number')
    assert df_no_indices.ww.index is None
    assert df_no_indices.ww.time_index is None

    df_with_indices = schema_df.ww.select(['Datetime', 'Integer'])
    assert df_with_indices.ww.index == 'id'
    assert df_with_indices.ww.time_index == 'signup_date'

    df_values = schema_df.ww.select(['FullName'])
    assert df_values.ww.name == schema_df.ww.name
    assert df_values.ww.columns['full_name'] == schema_df.ww.columns['full_name']


def test_select_semantic_tags(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(semantic_tags={'full_name': 'tag1',
                                     'email': ['tag2'],
                                     'age': ['numeric', 'tag2'],
                                     'phone_number': ['tag3', 'tag2'],
                                     'is_registered': 'category',
                                     },
                      time_index='signup_date')

    df_one_match = schema_df.ww.select('numeric')
    assert len(df_one_match.columns) == 2
    assert 'age' in df_one_match.columns
    assert 'id' in df_one_match.columns

    df_multiple_matches = schema_df.ww.select('tag2')
    assert len(df_multiple_matches.columns) == 3
    assert 'age' in df_multiple_matches.columns
    assert 'phone_number' in df_multiple_matches.columns
    assert 'email' in df_multiple_matches.columns

    df_multiple_tags = schema_df.ww.select(['numeric', 'time_index'])
    assert len(df_multiple_tags.columns) == 3
    assert 'id' in df_multiple_tags.columns
    assert 'age' in df_multiple_tags.columns
    assert 'signup_date' in df_multiple_tags.columns

    df_overlapping_tags = schema_df.ww.select(['numeric', 'tag2'])
    assert len(df_overlapping_tags.columns) == 4
    assert 'id' in df_overlapping_tags.columns
    assert 'age' in df_overlapping_tags.columns
    assert 'phone_number' in df_overlapping_tags.columns
    assert 'email' in df_overlapping_tags.columns

    df_common_tags = schema_df.ww.select(['category', 'numeric'])
    assert len(df_common_tags.columns) == 3
    assert 'id' in df_common_tags.columns
    assert 'is_registered' in df_common_tags.columns
    assert 'age' in df_common_tags.columns


def test_select_single_inputs(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={
                          'full_name': FullName,
                          'email': EmailAddress,
                          'phone_number': PhoneNumber,
                          'signup_date': Datetime(datetime_format='%Y-%m-%d')
                      },
                      semantic_tags={
                          'full_name': ['new_tag', 'tag2'],
                          'age': 'numeric',
                          'signup_date': 'date_of_birth'
                      })

    df_ltype_string = schema_df.ww.select('full_name')
    assert len(df_ltype_string.columns) == 1
    assert 'full_name' in df_ltype_string.columns

    df_ltype_obj = schema_df.ww.select(Integer)
    assert len(df_ltype_obj.columns) == 2
    assert 'age' in df_ltype_obj.columns
    assert 'id' in df_ltype_obj.columns

    df_tag_string = schema_df.ww.select('index')
    assert len(df_tag_string.columns) == 1
    assert 'id' in df_tag_string.columns

    df_tag_instantiated = schema_df.ww.select('Datetime')
    assert len(df_tag_instantiated.columns) == 1
    assert 'signup_date' in df_tag_instantiated.columns


def test_select_list_inputs(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={
                          'full_name': FullName,
                          'email': EmailAddress,
                          'phone_number': PhoneNumber,
                          'signup_date': Datetime(datetime_format='%Y-%m-%d'),
                      },
                      semantic_tags={
                          'full_name': ['new_tag', 'tag2'],
                          'age': 'numeric',
                          'signup_date': 'date_of_birth',
                          'email': 'tag2',
                          'is_registered': 'category'
                      })

    df_just_strings = schema_df.ww.select(['FullName', 'index', 'tag2', 'boolean'])
    assert len(df_just_strings.columns) == 4
    assert 'id' in df_just_strings.columns
    assert 'full_name' in df_just_strings.columns
    assert 'email' in df_just_strings.columns
    assert 'is_registered' in df_just_strings.columns

    df_mixed_selectors = schema_df.ww.select([FullName, 'index', 'time_index', Integer])
    assert len(df_mixed_selectors.columns) == 4
    assert 'id' in df_mixed_selectors.columns
    assert 'full_name' in df_mixed_selectors.columns
    assert 'signup_date' in df_mixed_selectors.columns
    assert 'age' in df_mixed_selectors.columns

    df_common_tags = schema_df.ww.select(['category', 'numeric', Boolean, Datetime])
    assert len(df_common_tags.columns) == 3
    assert 'is_registered' in df_common_tags.columns
    assert 'age' in df_common_tags.columns
    assert 'signup_date' in df_common_tags.columns


def test_select_semantic_tags_no_match(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={
                          'full_name': FullName,
                          'email': EmailAddress,
                          'phone_number': PhoneNumber,
                          'signup_date': Datetime(datetime_format='%Y-%m-%d'),
                      }, semantic_tags={
                          'full_name': ['new_tag', 'tag2'],
                          'age': 'numeric',
                          'signup_date': 'date_of_birth',
                          'email': 'tag2'
                      })

    assert len(schema_df.ww.select(['doesnt_exist']).columns) == 0

    df_multiple_unused = schema_df.ww.select(['doesnt_exist', 'boolean', 'category', PhoneNumber])
    assert len(df_multiple_unused.columns) == 2

    df_unused_ltype = schema_df.ww.select(['date_of_birth', 'doesnt_exist', ZIPCode, Integer])
    assert len(df_unused_ltype.columns) == 3


def test_select_repetitive(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={
                          'full_name': FullName,
                          'email': EmailAddress,
                          'phone_number': PhoneNumber,
                          'signup_date': Datetime(datetime_format='%Y-%m-%d'),
                      }, semantic_tags={
                          'full_name': ['new_tag', 'tag2'],
                          'age': 'numeric',
                          'signup_date': 'date_of_birth',
                          'email': 'tag2'
                      })
    df_repeat_tags = schema_df.ww.select(['new_tag', 'new_tag'])
    assert len(df_repeat_tags.columns) == 1
    assert set(df_repeat_tags.columns) == {'full_name'}

    df_repeat_ltypes = schema_df.ww.select(['PhoneNumber', PhoneNumber, 'phone_number'])
    assert len(df_repeat_ltypes.columns) == 1
    assert set(df_repeat_ltypes.columns) == {'phone_number'}


def test_accessor_set_index(sample_df):
    sample_df.ww.init()

    sample_df.ww.set_index('id')
    assert sample_df.ww.index == 'id'
    if isinstance(sample_df, pd.DataFrame):
        # underlying index not set for Dask/Koalas
        assert (sample_df.index == sample_df['id']).all()

    sample_df.ww.set_index('full_name')
    assert sample_df.ww.index == 'full_name'
    if isinstance(sample_df, pd.DataFrame):
        # underlying index not set for Dask/Koalas
        assert (sample_df.index == sample_df['full_name']).all()

    sample_df.ww.set_index(None)
    assert sample_df.ww.index is None
    if isinstance(sample_df, pd.DataFrame):
        # underlying index not set for Dask/Koalas
        # Check that underlying index doesn't get reset when Woodwork index is removed
        assert (sample_df.index == sample_df['full_name']).all()


def test_accessor_set_index_errors(sample_df):
    sample_df.ww.init()

    error = 'Specified index column `testing` not found in Schema.'
    with pytest.raises(LookupError, match=error):
        sample_df.ww.set_index('testing')

    if isinstance(sample_df, pd.DataFrame):
        # Index uniqueness not validate for Dask/Koalas
        error = "Index column must be unique"
        with pytest.raises(LookupError, match=error):
            sample_df.ww.set_index('age')


def test_set_types(sample_df):
    sample_df.ww.init(index='full_name', time_index='signup_date')

    original_df = sample_df.ww.copy()

    sample_df.ww.set_types()
    assert original_df.ww.schema == sample_df.ww.schema
    pd.testing.assert_frame_equal(to_pandas(original_df), to_pandas(sample_df))

    sample_df.ww.set_types(logical_types={'is_registered': 'Integer'})
    assert sample_df['is_registered'].dtype == 'Int64'

    sample_df.ww.set_types(semantic_tags={'signup_date': ['new_tag']},
                           logical_types={'full_name': 'Categorical'},
                           retain_index_tags=False)
    assert sample_df.ww.index is None
    assert sample_df.ww.time_index is None


def test_set_types_errors(sample_df):
    sample_df.ww.init(index='full_name')

    error = "String invalid is not a valid logical type"
    with pytest.raises(ValueError, match=error):
        sample_df.ww.set_types(logical_types={'id': 'invalid'})

    if isinstance(sample_df, pd.DataFrame):
        # Dask does not error on invalid type conversion until compute
        # Koalas does conversion and fills values with NaN
        error = 'Error converting datatype for email from type string ' \
            'to type float64. Please confirm the underlying data is consistent with ' \
            'logical type Double.'
        with pytest.raises(TypeConversionError, match=error):
            sample_df.ww.set_types(logical_types={'email': 'Double'})

    error = re.escape("Cannot add 'index' tag directly for column email. To set a column as the index, "
                      "use DataFrame.ww.set_index() instead.")
    with pytest.raises(ValueError, match=error):
        sample_df.ww.set_types(semantic_tags={'email': 'index'})


def test_pop(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(semantic_tags={'age': 'custom_tag'})
    original_schema = schema_df.ww.schema

    popped_series = schema_df.ww.pop('age')

    assert isinstance(popped_series, type(sample_df['age']))
    assert popped_series.ww.semantic_tags == {'custom_tag', 'numeric'}
    assert all(to_pandas(popped_series).values == [33, 25, 33, 57])
    assert popped_series.ww.logical_type == Integer

    assert 'age' not in schema_df.columns
    assert 'age' not in schema_df.ww.columns

    assert 'age' not in schema_df.ww.logical_types.keys()
    assert 'age' not in schema_df.ww.semantic_tags.keys()

    assert schema_df.ww.schema == original_schema._get_subset_schema(list(schema_df.columns))

    schema_df = sample_df.copy()
    schema_df.ww.init(
        name='table',
        logical_types={'age': Integer},
        semantic_tags={'age': 'custom_tag'},
        use_standard_tags=False)

    popped_series = schema_df.ww.pop('age')

    assert popped_series.ww.semantic_tags == {'custom_tag'}


def test_pop_index(sample_df):
    sample_df.ww.init(index='id', name='dt_name')
    assert sample_df.ww.index == 'id'
    id_col = sample_df.ww.pop('id')
    assert sample_df.ww.index is None
    assert 'index' in id_col.ww.semantic_tags


def test_pop_error(sample_df):
    sample_df.ww.init(
        name='table',
        logical_types={'age': Integer},
        semantic_tags={'age': 'custom_tag'},
        use_standard_tags=True)

    with pytest.raises(LookupError, match="Column with name missing not found in DataFrame"):
        sample_df.ww.pop("missing")


def test_accessor_drop(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init()

    single_input_df = schema_df.ww.drop('is_registered')
    assert len(single_input_df.ww.columns) == (len(schema_df.columns) - 1)
    assert 'is_registered' not in single_input_df.ww.columns
    assert to_pandas(schema_df).drop('is_registered', axis='columns').equals(to_pandas(single_input_df))

    list_input_df = schema_df.ww.drop(['is_registered'])
    assert len(list_input_df.ww.columns) == (len(schema_df.columns) - 1)
    assert 'is_registered' not in list_input_df.ww.columns
    assert to_pandas(schema_df).drop('is_registered', axis='columns').equals(to_pandas(list_input_df))
    # should be equal to the single input example above
    assert single_input_df.ww.schema == list_input_df.ww.schema
    assert to_pandas(single_input_df).equals(to_pandas(list_input_df))

    multiple_list_df = schema_df.ww.drop(['age', 'full_name', 'is_registered'])
    assert len(multiple_list_df.ww.columns) == (len(schema_df.columns) - 3)
    assert 'is_registered' not in multiple_list_df.ww.columns
    assert 'full_name' not in multiple_list_df.ww.columns
    assert 'age' not in multiple_list_df.ww.columns

    assert to_pandas(schema_df).drop(['is_registered', 'age', 'full_name'], axis='columns').equals(to_pandas(multiple_list_df))

    # Drop the same columns in a different order and confirm resulting DataFrame column order doesn't change
    different_order_df = schema_df.ww.drop(['is_registered', 'age', 'full_name'])
    assert different_order_df.ww.schema == multiple_list_df.ww.schema
    assert to_pandas(multiple_list_df).equals(to_pandas(different_order_df))


def test_accessor_drop_indices(sample_df):
    sample_df.ww.init(index='id', time_index='signup_date')
    assert sample_df.ww.index == 'id'
    assert sample_df.ww.time_index == 'signup_date'

    dropped_index_df = sample_df.ww.drop('id')
    assert 'id' not in dropped_index_df.ww.columns
    assert dropped_index_df.ww.index is None
    assert dropped_index_df.ww.time_index == 'signup_date'

    dropped_time_index_df = sample_df.ww.drop(['signup_date'])
    assert 'signup_date' not in dropped_time_index_df.ww.columns
    assert dropped_time_index_df.ww.time_index is None
    assert dropped_time_index_df.ww.index == 'id'


def test_accessor_drop_errors(sample_df):
    sample_df.ww.init()

    error = re.escape("['not_present'] not found in DataFrame")
    with pytest.raises(ValueError, match=error):
        sample_df.ww.drop('not_present')

    with pytest.raises(ValueError, match=error):
        sample_df.ww.drop(['age', 'not_present'])

    error = re.escape("['not_present1', 4] not found in DataFrame")
    with pytest.raises(ValueError, match=error):
        sample_df.ww.drop(['not_present1', 4])


def test_accessor_rename(sample_df):
    table_metadata = {'table_info': 'this is text'}
    id_description = 'the id of the row'
    sample_df.ww.init(index='id',
                      time_index='signup_date',
                      table_metadata=table_metadata,
                      column_descriptions={'id': id_description},
                      semantic_tags={'age': 'test_tag'},
                      logical_types={'age': Double})
    original_df = sample_df.ww.copy()

    new_df = sample_df.ww.rename({'age': 'birthday'})

    assert to_pandas(sample_df.rename(columns={'age': 'birthday'})).equals(to_pandas(new_df))
    # Confirm original dataframe hasn't changed
    assert to_pandas(sample_df).equals(to_pandas(original_df))
    assert sample_df.ww.schema == original_df.ww.schema

    assert original_df.columns.get_loc('age') == new_df.columns.get_loc('birthday')
    pd.testing.assert_series_equal(to_pandas(original_df['age']),
                                   to_pandas(new_df['birthday']),
                                   check_names=False)

    # confirm that metadata and descriptions are there
    assert new_df.ww.metadata == table_metadata
    assert new_df.ww.columns['id']['description'] == id_description

    old_col = sample_df.ww.columns['age']
    new_col = new_df.ww.columns['birthday']
    assert old_col['logical_type'] == new_col['logical_type']
    assert old_col['semantic_tags'] == new_col['semantic_tags']

    new_df = sample_df.ww.rename({'age': 'full_name', 'full_name': 'age'})

    pd.testing.assert_series_equal(to_pandas(original_df['age']),
                                   to_pandas(new_df['full_name']),
                                   check_names=False)
    pd.testing.assert_series_equal(to_pandas(original_df['full_name']),
                                   to_pandas(new_df['age']),
                                   check_names=False)

    assert original_df.columns.get_loc('age') == new_df.columns.get_loc('full_name')
    assert original_df.columns.get_loc('full_name') == new_df.columns.get_loc('age')


def test_accessor_rename_indices(sample_df):
    sample_df.ww.init(
        index='id',
        time_index='signup_date')

    renamed_df = sample_df.ww.rename({'id': 'renamed_index', 'signup_date': 'renamed_time_index'})
    assert 'id' not in renamed_df.columns
    assert 'signup_date' not in renamed_df.columns
    assert 'renamed_index' in renamed_df.columns
    assert 'renamed_time_index' in renamed_df.columns

    if isinstance(sample_df, pd.DataFrame):
        # underlying index not set for Dask/Koalas
        assert all(renamed_df.index == renamed_df['renamed_index'])

    assert renamed_df.ww.index == 'renamed_index'
    assert renamed_df.ww.time_index == 'renamed_time_index'


def test_accessor_schema_properties(sample_df):
    sample_df.ww.init(index='id',
                      time_index='signup_date')

    schema_properties = ['logical_types', 'semantic_tags', 'index', 'time_index']
    for schema_property in schema_properties:
        prop_from_accessor = getattr(sample_df.ww, schema_property)
        prop_from_schema = getattr(sample_df.ww.schema, schema_property)

        assert prop_from_accessor == prop_from_schema

        # Assumes we don't have setters for any of these attributes
        error = "can't set attribute"
        with pytest.raises(AttributeError, match=error):
            setattr(sample_df.ww, schema_property, 'new_value')


def test_sets_koalas_option_on_init(sample_df_koalas):
    if ks:
        ks.set_option('compute.ops_on_diff_frames', False)
        sample_df_koalas.ww.init()
        assert ks.get_option('compute.ops_on_diff_frames') is True


def test_maintain_column_order_on_type_changes(sample_df):
    sample_df.ww.init()
    schema_df = sample_df.ww.copy()

    schema_df.ww.set_types(logical_types={'email': 'Categorical', 'id': 'Double'},
                           semantic_tags={'age': 'tag', 'email': 'tag'})
    assert all(schema_df.columns == sample_df.columns)
    assert all(schema_df.ww.types.index == sample_df.ww.types.index)

    schema_df.ww.set_index('email')
    assert all(schema_df.columns == sample_df.columns)
    assert all(schema_df.ww.types.index == sample_df.ww.types.index)


def test_maintain_column_order_of_dataframe(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init()

    select_df = schema_df.ww.select([NaturalLanguage, Integer, Boolean, Datetime])
    assert all(schema_df.columns == select_df.columns)
    assert all(schema_df.ww.types.index == select_df.ww.types.index)

    renamed_df = schema_df.ww.rename({'email': 'renamed_1', 'id': 'renamed_2'})
    renamed_cols = ['renamed_2', 'full_name', 'renamed_1', 'phone_number', 'age', 'signup_date', 'is_registered']
    assert all(renamed_cols == renamed_df.columns)
    assert all(renamed_cols == renamed_df.ww.types.index)

    dropped_df = schema_df.ww.drop(['email', 'id', 'is_registered', 'age'])
    cols_left_over = ['full_name', 'phone_number', 'signup_date']
    assert all(cols_left_over == dropped_df.columns)
    assert all(cols_left_over == dropped_df.ww.types.index)


def test_maintain_column_order_of_input(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init()

    reversed_cols = list(schema_df.columns[::-1])

    if dd and not isinstance(sample_df, dd.DataFrame):
        iloc_df = schema_df.ww.iloc[:, list(range(len(schema_df.columns)))[::-1]]
        assert all(reversed_cols == iloc_df.columns)
        assert all(reversed_cols == iloc_df.ww.types.index)

    loc_df = schema_df.ww.loc[:, reversed_cols]
    assert all(reversed_cols == loc_df.columns)
    assert all(reversed_cols == loc_df.ww.types.index)

    getitem_df = schema_df.ww[reversed_cols]
    assert all(reversed_cols == getitem_df.columns)
    assert all(reversed_cols == getitem_df.ww.types.index)


def test_maintain_column_order_disordered_schema(sample_df):
    sample_df.ww.init()
    column_order = list(sample_df.columns)

    scramble_df = sample_df.ww.copy()
    id_col = scramble_df.ww.columns.pop('id')
    scramble_df.ww.columns['id'] = id_col
    assert list(scramble_df.ww.columns.keys()) != column_order

    assert scramble_df.ww.schema == sample_df.ww.schema
    assert all(scramble_df.columns == column_order)
    assert all(scramble_df.ww.types.index == column_order)

    sample_df.ww.init(schema=scramble_df.ww.schema)
    assert all(sample_df.columns == column_order)
    assert all(sample_df.ww.types.index == column_order)


def test_accessor_types(sample_df):
    sample_df.ww.init()

    returned_types = sample_df.ww.types
    assert isinstance(returned_types, pd.DataFrame)
    assert 'Physical Type' in returned_types.columns
    assert 'Logical Type' in returned_types.columns
    assert 'Semantic Tag(s)' in returned_types.columns
    assert returned_types.shape[1] == 3
    assert len(returned_types.index) == len(sample_df.columns)

    string_dtype = 'string'
    boolean_dtype = 'boolean'
    int_dtype = 'Int64'

    correct_physical_types = {
        'id': int_dtype,
        'full_name': string_dtype,
        'email': string_dtype,
        'phone_number': string_dtype,
        'age': int_dtype,
        'signup_date': 'datetime64[ns]',
        'is_registered': boolean_dtype,
    }
    correct_physical_types = pd.Series(list(correct_physical_types.values()),
                                       index=list(correct_physical_types.keys()))
    assert correct_physical_types.equals(returned_types['Physical Type'])

    correct_logical_types = {
        'id': Integer,
        'full_name': NaturalLanguage,
        'email': NaturalLanguage,
        'phone_number': NaturalLanguage,
        'age': Integer,
        'signup_date': Datetime,
        'is_registered': Boolean,
    }
    correct_logical_types = pd.Series(list(correct_logical_types.values()),
                                      index=list(correct_logical_types.keys()))
    assert correct_logical_types.equals(returned_types['Logical Type'])

    correct_semantic_tags = {
        'id': "['numeric']",
        'full_name': "[]",
        'email': "[]",
        'phone_number': "[]",
        'age': "['numeric']",
        'signup_date': "[]",
        'is_registered': "[]",
    }
    correct_semantic_tags = pd.Series(list(correct_semantic_tags.values()),
                                      index=list(correct_semantic_tags.keys()))
    assert correct_semantic_tags.equals(returned_types['Semantic Tag(s)'])


def test_accessor_repr(small_df):
    small_df.ww.init()

    table_repr = repr(small_df.ww)
    expected_repr = '                         Physical Type Logical Type Semantic Tag(s)\nColumn                                                             \nsample_datetime_series  datetime64[ns]     Datetime              []'
    assert table_repr == expected_repr

    table_html_repr = small_df.ww._repr_html_()
    expected_repr = '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Physical Type</th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>sample_datetime_series</th>\n      <td>datetime64[ns]</td>\n      <td>Datetime</td>\n      <td>[]</td>\n    </tr>\n  </tbody>\n</table>'
    assert table_html_repr == expected_repr


def test_accessor_repr_empty(empty_df):
    empty_df.ww.init()

    assert repr(empty_df.ww) == 'Empty DataFrame\nColumns: [Physical Type, Logical Type, Semantic Tag(s)]\nIndex: []'
    assert empty_df.ww._repr_html_() == '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Physical Type</th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n  </tbody>\n</table>'
