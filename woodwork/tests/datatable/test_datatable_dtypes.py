import re

import numpy as np
import pandas as pd
import pytest

from woodwork import DataTable
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


def test_datatable_physical_types(sample_df):
    dt = DataTable(sample_df)
    assert isinstance(dt.physical_types, dict)
    assert set(dt.physical_types.keys()) == set(sample_df.columns)
    for k, v in dt.physical_types.items():
        assert isinstance(k, str)
        assert v == sample_df[k].dtype


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
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
            assert dt.columns[column_name].logical_type == logical_type
            assert dt.columns[column_name].dtype == logical_type.pandas_dtype
            assert dt.to_dataframe()[column_name].dtype == logical_type.pandas_dtype


def test_sets_category_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series(['a', 'b', 'c'], name=column_name)
    series = series.astype('object')
    logical_types = [
        Categorical,
        CountryCode,
        Ordinal(order=['a', 'b', 'c']),
        SubRegionCode,
        ZIPCode,
    ]

    for logical_type in logical_types:
        ltypes = {
            column_name: NaturalLanguage,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        dt = dt.set_types(logical_types={column_name: logical_type})
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_dataframe()[column_name].dtype == logical_type.pandas_dtype


def test_sets_object_dtype_on_init(latlong_df):
    for column_name in latlong_df.columns:
        ltypes = {
            column_name: LatLong,
        }
        dt = DataTable(latlong_df.loc[:, [column_name]], logical_types=ltypes)
        assert dt.columns[column_name].logical_type == LatLong
        assert dt.columns[column_name].dtype == LatLong.pandas_dtype
        assert dt.to_dataframe()[column_name].dtype == LatLong.pandas_dtype


def test_sets_object_dtype_on_update(latlong_df):
    for column_name in latlong_df.columns:
        ltypes = {
            column_name: NaturalLanguage
        }
        dt = DataTable(latlong_df.loc[:, [column_name]], logical_types=ltypes)
        dt = dt.set_types(logical_types={column_name: LatLong})
        assert dt.columns[column_name].logical_type == LatLong
        assert dt.columns[column_name].dtype == LatLong.pandas_dtype
        assert dt.to_dataframe()[column_name].dtype == LatLong.pandas_dtype


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
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
            assert dt.columns[column_name].logical_type == logical_type
            assert dt.columns[column_name].dtype == logical_type.pandas_dtype
            assert dt.to_dataframe()[column_name].dtype == logical_type.pandas_dtype


def test_sets_string_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series(['a', 'b', 'c'], name=column_name)
    series = series.astype('object')
    logical_types = [
        Filepath,
        FullName,
        IPAddress,
        NaturalLanguage,
        PhoneNumber,
        URL,
    ]

    for logical_type in logical_types:
        ltypes = {
            column_name: Categorical,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        dt = dt.set_types(logical_types={column_name: logical_type})
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_dataframe()[column_name].dtype == logical_type.pandas_dtype


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
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_dataframe()[column_name].dtype == logical_type.pandas_dtype


def test_sets_boolean_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series([0, 1, 0], name=column_name)
    series = series.astype('object')
    ltypes = {
        column_name: Integer,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    dt = dt.set_types(logical_types={column_name: Boolean})
    assert dt.columns[column_name].logical_type == Boolean
    assert dt.columns[column_name].dtype == Boolean.pandas_dtype
    assert dt.to_dataframe()[column_name].dtype == Boolean.pandas_dtype


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
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_dataframe()[column_name].dtype == logical_type.pandas_dtype


def test_sets_int64_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series([1.0, 2.0, 1.0], name=column_name)
    series = series.astype('object')
    logical_types = [Integer]

    for logical_type in logical_types:
        ltypes = {
            column_name: Double,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        dt = dt.set_types(logical_types={column_name: logical_type})
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_dataframe()[column_name].dtype == logical_type.pandas_dtype


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
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_dataframe()[column_name].dtype == logical_type.pandas_dtype


def test_sets_float64_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series([0, 1, 0], name=column_name)
    series = series.astype('object')
    ltypes = {
        column_name: Integer,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    dt = dt.set_types(logical_types={column_name: Double})
    assert dt.columns[column_name].logical_type == Double
    assert dt.columns[column_name].dtype == Double.pandas_dtype
    assert dt.to_dataframe()[column_name].dtype == Double.pandas_dtype


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
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_dataframe()[column_name].dtype == logical_type.pandas_dtype


def test_sets_datetime_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name)
    series = series.astype('object')
    ltypes = {
        column_name: NaturalLanguage,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    dt = dt.set_types(logical_types={column_name: Datetime})
    assert dt.columns[column_name].logical_type == Datetime
    assert dt.columns[column_name].dtype == Datetime.pandas_dtype
    assert dt.to_dataframe()[column_name].dtype == Datetime.pandas_dtype


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
        DataTable(pd.DataFrame(series), logical_types=ltypes)

    # Cannot cast Datetime to Double
    series = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name)
    ltypes = {
        column_name: Datetime,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    err_msg = 'Error converting datatype for column test_series from type datetime64[ns] to type ' \
        'float64. Please confirm the underlying data is consistent with logical type Double.'
    with pytest.raises(TypeError, match=re.escape(err_msg)):
        dt.set_types(logical_types={column_name: Double})

    # Cannot cast invalid strings to integers
    series = pd.Series(['1', 'two', '3'], name=column_name)
    ltypes = {
        column_name: Integer,
    }
    err_msg = 'Error converting datatype for column test_series from type object to type ' \
        'Int64. Please confirm the underlying data is consistent with logical type Integer.'
    with pytest.raises(TypeError, match=err_msg):
        DataTable(pd.DataFrame(series), logical_types=ltypes)


def test_int_dtype_inference_on_init():
    df = pd.DataFrame({
        'ints_no_nans': pd.Series([1, 2]),
        'ints_nan': pd.Series([1, np.nan]),
        'ints_NA': pd.Series([1, pd.NA]),
        'ints_NA_specified': pd.Series([1, pd.NA], dtype='Int64')})
    df_from_dt = DataTable(df).to_dataframe()

    assert df_from_dt['ints_no_nans'].dtype == 'Int64'
    assert df_from_dt['ints_nan'].dtype == 'float64'
    assert df_from_dt['ints_NA'].dtype == 'category'
    assert df_from_dt['ints_NA_specified'].dtype == 'Int64'


def test_bool_dtype_inference_on_init():
    df = pd.DataFrame({
        'bools_no_nans': pd.Series([True, False]),
        'bool_nan': pd.Series([True, np.nan]),
        'bool_NA': pd.Series([True, pd.NA]),
        'bool_NA_specified': pd.Series([True, pd.NA], dtype="boolean")})
    df_from_dt = DataTable(df).to_dataframe()

    assert df_from_dt['bools_no_nans'].dtype == 'boolean'
    assert df_from_dt['bool_nan'].dtype == 'category'
    assert df_from_dt['bool_NA'].dtype == 'category'
    assert df_from_dt['bool_NA_specified'].dtype == 'boolean'


def test_str_dtype_inference_on_init():
    df = pd.DataFrame({
        'str_no_nans': pd.Series(['a', 'b']),
        'str_nan': pd.Series(['a', np.nan]),
        'str_NA': pd.Series(['a', pd.NA]),
        'str_NA_specified': pd.Series([1, pd.NA], dtype="string"),
        'long_str_NA_specified': pd.Series(['this is a very long sentence inferred as a string', pd.NA], dtype="string"),
        'long_str_NA': pd.Series(['this is a very long sentence inferred as a string', pd.NA])
    })
    df_from_dt = DataTable(df).to_dataframe()

    assert df_from_dt['str_no_nans'].dtype == 'category'
    assert df_from_dt['str_nan'].dtype == 'category'
    assert df_from_dt['str_NA'].dtype == 'category'
    assert df_from_dt['str_NA_specified'].dtype == 'category'
    assert df_from_dt['long_str_NA_specified'].dtype == 'string'
    assert df_from_dt['long_str_NA'].dtype == 'string'


def test_float_dtype_inference_on_init():
    df = pd.DataFrame({
        'floats_no_nans': pd.Series([1.1, 2.2]),
        'floats_nan': pd.Series([1.1, np.nan]),
        'floats_NA': pd.Series([1.1, pd.NA]),
        'floats_nan_specified': pd.Series([1.1, np.nan], dtype='float')})
    df_from_dt = DataTable(df).to_dataframe()

    assert df_from_dt['floats_no_nans'].dtype == 'float64'
    assert df_from_dt['floats_nan'].dtype == 'float64'
    assert df_from_dt['floats_NA'].dtype == 'category'
    assert df_from_dt['floats_nan_specified'].dtype == 'float64'


def test_datetime_dtype_inference_on_init():
    df = pd.DataFrame({
        'date_no_nans': pd.Series([pd.to_datetime('2020-09-01')] * 2),
        'date_nan': pd.Series([pd.to_datetime('2020-09-01'), np.nan]),
        'date_NA': pd.Series([pd.to_datetime('2020-09-01'), pd.NA]),
        'date_NaT': pd.Series([pd.to_datetime('2020-09-01'), pd.NaT]),
        'date_NA_specified': pd.Series([pd.to_datetime('2020-09-01'), pd.NA], dtype='datetime64[ns]')})
    df_from_dt = DataTable(df).to_dataframe()

    assert df_from_dt['date_no_nans'].dtype == 'datetime64[ns]'
    assert df_from_dt['date_nan'].dtype == 'datetime64[ns]'
    assert df_from_dt['date_NA'].dtype == 'datetime64[ns]'
    assert df_from_dt['date_NaT'].dtype == 'datetime64[ns]'
    assert df_from_dt['date_NA_specified'].dtype == 'datetime64[ns]'


def test_timedelta_dtype_inference_on_init():
    df = pd.DataFrame({
        'delta_no_nans': (pd.Series([pd.to_datetime('2020-09-01')] * 2) - pd.to_datetime('2020-07-01')),
        'delta_nan': (pd.Series([pd.to_datetime('2020-09-01'), np.nan]) - pd.to_datetime('2020-07-01')),
        'delta_NaT': (pd.Series([pd.to_datetime('2020-09-01'), pd.NaT]) - pd.to_datetime('2020-07-01')),
        'delta_NA_specified': (pd.Series([pd.to_datetime('2020-09-01'), pd.NA], dtype='datetime64[ns]') - pd.to_datetime('2020-07-01')),
    })
    df_from_dt = DataTable(df).to_dataframe()

    assert df_from_dt['delta_no_nans'].dtype == 'timedelta64[ns]'
    assert df_from_dt['delta_nan'].dtype == 'timedelta64[ns]'
    assert df_from_dt['delta_NaT'].dtype == 'timedelta64[ns]'
    assert df_from_dt['delta_NA_specified'].dtype == 'timedelta64[ns]'
