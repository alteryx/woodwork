import re

import numpy as np
import pandas as pd
import pytest

from woodwork.logical_types import (
    URL,
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
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
    _check_unique_column_names
)
from woodwork.tests.testing_utils import to_pandas
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def xfail_dask_and_koalas(df):
    if dd and isinstance(df, dd.DataFrame) or ks and isinstance(df, ks.DataFrame):
        pytest.xfail('Dask and Koalas Accessors not yet supported.')


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
    xfail_dask_and_koalas(sample_df)

    assert sample_df.ww.schema is None
    sample_df.ww.init()
    assert isinstance(sample_df.ww.schema, Schema)


def test_accessor_separation_of_params(sample_df):
    xfail_dask_and_koalas(sample_df)
    # mix up order of acccessor and schema params
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_name', index='id', semantic_tags={'id': 'test_tag'}, time_index='signup_date')

    assert schema_df.ww.semantic_tags['id'] == {'index', 'test_tag'}
    assert schema_df.ww.index == 'id'
    assert schema_df.ww.time_index == 'signup_date'
    assert schema_df.ww.name == 'test_name'


def test_accessor_getattr(sample_df):
    xfail_dask_and_koalas(sample_df)

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


def test_accessor_equality_with_schema(sample_df, sample_column_names, sample_inferred_logical_types):
    xfail_dask_and_koalas(sample_df)
    comparison_schema = Schema(sample_column_names, sample_inferred_logical_types)

    schema_df = sample_df.copy()
    schema_df.ww.init()

    # eq not implemented on Accessor class, so Schema's eq is called
    assert schema_df.ww.__eq__(comparison_schema)

    # Since there's a repr on Accessor, it gets called
    assert schema_df.ww._repr_html_() == comparison_schema._repr_html_()
    assert schema_df.ww.__repr__() == comparison_schema.__repr__()

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
    xfail_dask_and_koalas(time_index_df)

    time_index_df.ww.init(name='schema',
                          index='id',
                          time_index='times')

    assert time_index_df.ww.name == 'schema'
    assert time_index_df.ww.index == 'id'
    assert time_index_df.ww.time_index == 'times'
    assert time_index_df.ww.columns[time_index_df.ww.time_index]['logical_type'] == Datetime


def test_accessor_init_with_numeric_datetime_time_index(time_index_df):
    xfail_dask_and_koalas(time_index_df)
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints', logical_types={'ints': Datetime})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        time_index_df.ww.init(name='schema', time_index='strs', logical_types={'strs': Datetime})

    assert schema_df.ww.time_index == 'ints'
    assert schema_df['ints'].dtype == 'datetime64[ns]'


def test_accessor_with_numeric_time_index(time_index_df):
    xfail_dask_and_koalas(time_index_df)
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

    # --> add back when schema updates are implemented
    # # Change time index to normal datetime time index
    # schema = schema.set_time_index('times')
    # date_col = schema['ints']
    # assert schema.time_index == 'times'
    # assert date_col.logical_type == Double
    # assert date_col.semantic_tags == {'numeric'}

    # Set numeric time index after init
    # schema = Schema(time_index_df, logical_types={'ints': 'Double'})
    # schema = schema.set_time_index('ints')
    # date_col = schema['ints']
    # assert schema.time_index == 'ints'
    # assert date_col.logical_type == Double
    # assert date_col.semantic_tags == {'time_index', 'numeric'}


def test_accessor_init_with_invalid_string_time_index(sample_df):
    xfail_dask_and_koalas(sample_df)
    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        sample_df.ww.init(name='schema', time_index='full_name')


def test_accessor_init_with_string_logical_types(sample_df):
    xfail_dask_and_koalas(sample_df)
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
            assert df.ww.columns[column_name]['dtype'] == logical_type.pandas_dtype
            assert df[column_name].dtype == logical_type.pandas_dtype


def test_sets_object_dtype_on_init(latlong_df):
    xfail_dask_and_koalas(latlong_df)
    for column_name in latlong_df.columns:
        ltypes = {
            column_name: LatLong,
        }
        df = latlong_df.loc[:, [column_name]]
        df.ww.init(logical_types=ltypes)
        assert df.ww.columns[column_name]['logical_type'] == LatLong
        assert df.ww.columns[column_name]['dtype'] == LatLong.pandas_dtype
        assert df[column_name].dtype == LatLong.pandas_dtype

        assert df[column_name].iloc[-1] == (3, 4)


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
            assert df.ww.columns[column_name]['dtype'] == logical_type.pandas_dtype
            assert df[column_name].dtype == logical_type.pandas_dtype


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
        assert df.ww.columns[column_name]['dtype'] == logical_type.pandas_dtype
        assert df[column_name].dtype == logical_type.pandas_dtype


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
            assert df.ww.columns[column_name]['dtype'] == logical_type.pandas_dtype
            assert df[column_name].dtype == logical_type.pandas_dtype


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
        assert df.ww.columns[column_name]['dtype'] == logical_type.pandas_dtype
        assert df[column_name].dtype == logical_type.pandas_dtype


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
        assert df.ww.columns[column_name]['dtype'] == logical_type.pandas_dtype
        assert df[column_name].dtype == logical_type.pandas_dtype


def test_invalid_dtype_casting():
    column_name = 'test_series'

    # Cannot cast a column with pd.NA to Double
    series = pd.Series([1.1, pd.NA, 3], name=column_name)
    ltypes = {
        column_name: Double,
    }
    err_msg = 'Error converting datatype for column test_series from type object to type ' \
        'float64. Please confirm the underlying data is consistent with logical type Double.'
    with pytest.raises(TypeError, match=err_msg):
        pd.DataFrame(series).ww.init(logical_types=ltypes)

    # --> add back when schema updates are implemented
    # # Cannot cast Datetime to Double
    # series = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name)
    # ltypes = {
    #     column_name: Datetime,
    # }
    # schema = Schema(pd.DataFrame(series), logical_types=ltypes)
    # err_msg = 'Error converting datatype for column test_series from type datetime64[ns] to type ' \
    #     'float64. Please confirm the underlying data is consistent with logical type Double.'
    # with pytest.raises(TypeError, match=re.escape(err_msg)):
    #     schema.set_types(logical_types={column_name: Double})

    # Cannot cast invalid strings to integers
    series = pd.Series(['1', 'two', '3'], name=column_name)
    ltypes = {
        column_name: Integer,
    }
    err_msg = 'Error converting datatype for column test_series from type object to type ' \
        'Int64. Please confirm the underlying data is consistent with logical type Integer.'
    with pytest.raises(TypeError, match=err_msg):
        pd.DataFrame(series).ww.init(logical_types=ltypes)


def test_make_index(sample_df):
    xfail_dask_and_koalas(sample_df)

    schema_df = sample_df.copy()
    schema_df.ww.init(index='new_index', make_index=True)

    assert schema_df.ww.index == 'new_index'
    assert 'new_index' in schema_df.ww.columns
    assert 'new_index' in schema_df.ww.columns
    assert to_pandas(schema_df)['new_index'].unique
    assert to_pandas(schema_df['new_index']).is_monotonic
    assert 'index' in schema_df.ww.columns['new_index']['semantic_tags']


def test_underlying_index_no_index(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    assert type(sample_df.index) == pd.RangeIndex

    schema_df = sample_df.copy()
    schema_df.ww.init()
    assert type(schema_df.index) == pd.RangeIndex

    sample_df = sample_df.sort_values('full_name')
    assert type(sample_df.index) == pd.Int64Index
    sample_df.ww.init()
    assert type(sample_df.index) == pd.RangeIndex


def test_underlying_index(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    specified_index = pd.Index

    schema_df = sample_df.copy()
    schema_df.ww.init(index='full_name')
    assert schema_df.index.name is None
    assert (schema_df.index == ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown', 'Ms. Paige Turner']).all()
    assert type(schema_df.index) == specified_index

    # --> add back when schema updates are implemented
    # schema = Schema(sample_df.copy())
    # schema = schema.set_index('full_name')
    # assert (schema._dataframe.index == schema.to_dataframe()['full_name']).all()
    # assert schema._dataframe.index.name is None
    # assert type(schema._dataframe.index) == specified_index
    # assert type(schema.to_dataframe().index) == specified_index

    # schema.index = 'id'
    # assert (schema._dataframe.index == [0, 1, 2, 3]).all()
    # assert schema._dataframe.index.name is None
    # assert type(schema._dataframe.index) == specified_index
    # assert type(schema.to_dataframe().index) == specified_index

    # # test removing index removes the dataframe's index
    # schema.index = None
    # assert type(schema._dataframe.index) == unspecified_index
    # assert type(schema.to_dataframe().index) == unspecified_index

    schema_df = sample_df.copy()
    schema_df.ww.init(index='made_index', make_index=True)
    assert (schema_df.index == [0, 1, 2, 3]).all()
    assert schema_df.index.name is None
    assert type(schema_df.index) == specified_index

    # --> add back when schema updates are implemented
    # schema_dropped = schema.drop('made_index')
    # assert 'made_index' not in schema_dropped.columns
    # assert 'made_index' not in schema_dropped._dataframe.columns
    # assert type(schema_dropped._dataframe.index) == unspecified_index
    # assert type(schema_dropped.to_dataframe().index) == unspecified_index


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

    # --> add back when schema updates are implemented
    # dc = DataColumn(sample_series, logical_type="NaturalLanguage")
    # new_dc = dc.set_logical_type(ordinal_with_order)
    # assert isinstance(new_dc.logical_type, Ordinal)
    # assert new_dc.logical_type.order == ['a', 'b', 'c']


def test_ordinal_with_incomplete_ranking(sample_series):
    if (ks and isinstance(sample_series, ks.Series)) or (dd and isinstance(sample_series, dd.Series)):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not supported')

    ordinal_incomplete_order = Ordinal(order=['a', 'b'])
    error_msg = re.escape("Ordinal column sample_series contains values that are not "
                          "present in the order values provided: ['c']")
    with pytest.raises(ValueError, match=error_msg):
        schema_df = pd.DataFrame(sample_series)
        schema_df.ww.init(logical_types={'sample_series': ordinal_incomplete_order})


def test_ordinal_with_nan_values():
    nan_df = pd.DataFrame(pd.Series(['a', 'b', np.nan, 'a'], name='nan_series'))
    ordinal_with_order = Ordinal(order=['a', 'b'])
    nan_df.ww.init(logical_types={'nan_series': ordinal_with_order})

    column_logical_type = nan_df.ww.logical_types['nan_series']
    assert isinstance(column_logical_type, Ordinal)
    assert column_logical_type.order == ['a', 'b']


def test_accessor_with_falsy_column_names(falsy_names_df):
    if dd and isinstance(falsy_names_df, dd.DataFrame):
        pytest.xfail('Dask DataTables cannot handle integer column names')
    xfail_dask_and_koalas(falsy_names_df)

    schema_df = falsy_names_df.copy()
    schema_df.ww.init(index=0, time_index='')
    assert schema_df.ww.index == 0
    assert schema_df.ww.time_index == ''

    # --> add back in when series accessor is implemented
    # for col_name in falsy_names_df.columns:
    #     dc = dt[col_name]
    #     assert dc.name == col_name
    #     assert dc._series.name == col_name

    # --> add back in once schema updates are implemented
    # dt.time_index = None
    # assert dt.time_index is None

    # --> add back in once pop and rename are implemented on the accessor
    # popped_col = dt.pop('')
    # dt[''] = popped_col
    # assert dt[''].name == ''
    # assert dt['']._series.name == ''

    # dt = dt.rename({'': 'col_with_name'})
    # assert '' not in dt.columns
    # assert 'col_with_name' in dt.columns


def test_accessor_repr(sample_df, sample_column_names, sample_inferred_logical_types):
    xfail_dask_and_koalas(sample_df)

    schema = Schema(sample_column_names, sample_inferred_logical_types)
    sample_df.ww.init()

    assert repr(schema) == repr(sample_df.ww)
