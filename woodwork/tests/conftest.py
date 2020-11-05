import numpy as np
import pandas as pd
import pytest

import woodwork as ww
from woodwork.logical_types import Boolean, Categorical, Integer


@pytest.fixture(params=['sample_df_pandas', 'sample_df_dask'])
def sample_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_df_pandas():
    return pd.DataFrame({
        'id': range(4),
        'full_name': ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown', 'Ms. Paige Turner'],
        'email': ['john.smith@example.com', np.nan, 'team@featuretools.com', 'junk@example.com'],
        'phone_number': ['5555555555', '555-555-5555', '1-(555)-555-5555', '555-555-5555'],
        'age': [33, 25, 33, 57],
        'signup_date': [pd.to_datetime('2020-09-01')] * 4,
        'is_registered': [True, False, True, True],
    })


@pytest.fixture()
def sample_df_dask(sample_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(sample_df_pandas, npartitions=2)


@pytest.fixture(params=['sample_series_pandas', 'sample_series_dask'])
def sample_series(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_series_pandas():
    return pd.Series(['a', 'b', 'c', 'a'], name='sample_series').astype('object')


@pytest.fixture()
def sample_series_dask(sample_series_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(sample_series_pandas, npartitions=2)


@pytest.fixture(params=['sample_datetime_series_pandas', 'sample_datetime_series_dask'])
def sample_datetime_series(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_datetime_series_pandas():
    return pd.Series([pd.to_datetime('2020-09-01')] * 4, name='sample_datetime_series').astype('object')


@pytest.fixture()
def sample_datetime_series_dask(sample_datetime_series_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(sample_datetime_series_pandas, npartitions=2)


@pytest.fixture()
def time_index_df_pandas():
    return pd.DataFrame({
        'id': [0, 1, 2, 3],
        'times': ['2019-01-01', '2019-01-02', '2019-01-03', pd.NA],
        'ints': [1, 2, 3, 4],
        'strs': ['1', '2', '3', '4'],
        'letters': ['a', 'b', 'c', 'd'],
        'bools': [True, False, False, True]
    })


@pytest.fixture()
def time_index_df_dask(time_index_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(time_index_df_pandas, npartitions=2)


@pytest.fixture(params=['time_index_df_pandas', 'time_index_df_dask'])
def time_index_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def numeric_time_index_df_pandas():
    return pd.DataFrame({
        'whole_numbers': pd.Series([1, 2, 3, 4], dtype='Int64'),
        'floats': pd.Series([1, 2, 3, 4], dtype='float'),
        'ints': pd.Series([1, -2, 3, 4], dtype='Int64'),
        'with_null': pd.Series([1, 2, pd.NA, 4], dtype='Int64'),
    })


@pytest.fixture()
def numeric_time_index_df_dask(numeric_time_index_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(numeric_time_index_df_pandas, npartitions=2)


@pytest.fixture(params=['numeric_time_index_df_pandas', 'numeric_time_index_df_dask'])
def numeric_time_index_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def describe_df_pandas():
    index_data = [0, 1, 2, 3, 4, 5, 6, 7]
    boolean_data = [True, False, True, True, False, True, np.nan, True]
    category_data = ['red', 'blue', 'red', np.nan, 'red', 'blue', 'red', 'yellow']
    datetime_data = pd.to_datetime(['2020-01-01',
                                    '2020-02-01',
                                    '2020-01-01 08:00',
                                    '2020-02-02 16:00',
                                    '2020-02-02 18:00',
                                    pd.NaT,
                                    '2020-02-01',
                                    '2020-01-02'])
    formatted_datetime_data = pd.Series(['2020~01~01',
                                         '2020~02~01',
                                         '2020~03~01',
                                         '2020~02~02',
                                         '2020~03~02',
                                         pd.NaT,
                                         '2020~02~01',
                                         '2020~01~02'])
    numeric_data = pd.Series([10, 20, 17, 32, np.nan, 1, 56, 10])
    natural_language_data = [
        'This is a natural language sentence',
        'Duplicate sentence.',
        'This line has numbers in it 000123.',
        'How about some symbols?!',
        'This entry contains two sentences. Second sentence.',
        'Duplicate sentence.',
        np.nan,
        'I am the last line',
    ]
    timedelta_data = datetime_data - pd.Timestamp('2020-01-01')

    return pd.DataFrame({
        'index_col': index_data,
        'boolean_col': boolean_data,
        'category_col': category_data,
        'datetime_col': datetime_data,
        'formatted_datetime_col': formatted_datetime_data,
        'numeric_col': numeric_data,
        'natural_language_col': natural_language_data,
        'timedelta_col': timedelta_data,
    })


@pytest.fixture()
def describe_df_dask(describe_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(describe_df_pandas, npartitions=2)


@pytest.fixture(params=['describe_df_pandas', 'describe_df_dask'])
def describe_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def df_same_mi_pandas():
    return pd.DataFrame({
        'ints': pd.Series([2, pd.NA, 5, 2], dtype='Int64'),
        'floats': pd.Series([1, None, 100, 1]),
        'nans': pd.Series([None, None, None, None]),
        'nat_lang': pd.Series(['this is a very long sentence inferred as a string', None, 'test', 'test']),
        'date': pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'])
    })


@pytest.fixture()
def df_same_mi_dask(df_same_mi_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(df_same_mi_pandas, npartitions=2)


@pytest.fixture(params=['df_same_mi_pandas', 'df_same_mi_dask'])
def df_same_mi(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def df_mi_pandas():
    return pd.DataFrame({
        'ints': pd.Series([1, 2, 3]),
        'bools': pd.Series([True, False, True]),
        'strs2': pd.Series(['bye', 'hi', 'bye']),
        'strs': pd.Series(['hi', 'hi', 'hi'])
    })


@pytest.fixture()
def df_mi_dask(df_mi_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(df_mi_pandas, npartitions=1)


@pytest.fixture(params=['df_mi_pandas', 'df_mi_dask'])
def df_mi(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def categorical_df():
    return pd.DataFrame({
        'ints': pd.Series([1, 2, 3, 2]),
        'categories1': pd.Series([1, 100, 1, 100, 200, 200, 200, 200, 3]),
        'bools': pd.Series([True, False, True, False]),
        'categories2': pd.Series(['test', 'test2', 'test2', 'test']),
        'categories3': pd.Series(['test', 'test', 'test', np.nan]),
    })


@pytest.fixture()
def categorical_log_types():
    return {
        'ints': Integer,
        'categories1': Categorical,
        'bools': Boolean,
        'categories2': Categorical,
        'categories3': Categorical,
    }


@pytest.fixture()
def categorical_dd(categorical_df):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(categorical_df, npartitions=2)


@pytest.fixture()
def categorical_pandas_dd_list(categorical_df, categorical_dd, categorical_log_types):
    return [
        ww.DataTable(categorical_df, logical_types=categorical_log_types),
        ww.DataTable(categorical_dd, logical_types=categorical_log_types)
    ]
