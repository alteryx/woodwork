import numpy as np
import pandas as pd
import pytest

from woodwork.logical_types import (
    Boolean,
    BooleanNullable,
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    Integer,
    IntegerNullable,
    PersonFullName,
    PhoneNumber,
    Unknown
)
from woodwork.utils import import_or_none


def pd_to_dask(series):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(series, npartitions=2)


def pd_to_koalas(series):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(series)


@pytest.fixture(scope='session', autouse=True)
def spark_session():
    pyspark = import_or_none('pyspark.sql')

    if pyspark:
        spark = pyspark.SparkSession.builder \
            .master("local[*]") \
            .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=True") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .getOrCreate()

        return spark


@pytest.fixture(params=['sample_df_pandas', 'sample_df_dask', 'sample_df_koalas'])
def sample_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_df_pandas():
    return pd.DataFrame({
        'id': range(4),
        'full_name': ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown', 'Ms. Paige Turner'],
        'email': ['john.smith@example.com', np.nan, 'team@featuretools.com', 'junk@example.com'],
        'phone_number': ['5555555555', '555-555-5555', '1-(555)-555-5555', '555-555-5555'],
        'age': pd.Series([pd.NA, 33, 33, 57], dtype='Int64'),
        'signup_date': [pd.to_datetime('2020-09-01')] * 4,
        'is_registered': pd.Series([pd.NA, False, True, True], dtype='boolean'),
        'double': [0, 1, 0.0001, -5.4321],
        'double_with_nan': [np.nan, -123.45, 0.0001, 1],
        'integer': [-1, 4, 9, 25],
        'nullable_integer': pd.Series([pd.NA, -36, 49, 64], dtype='Int64'),
        'boolean': [True, False, False, True],
        'categorical': pd.Series(["a", "b", "c", "a"], dtype="category"),
        'datetime_with_NaT': [pd.to_datetime('2020-09-01')] * 3 + [pd.NaT],
    })


@pytest.fixture()
def sample_df_dask(sample_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(sample_df_pandas, npartitions=2)


@pytest.fixture()
def sample_df_koalas(sample_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(sample_df_pandas)


@pytest.fixture(params=['sample_unsorted_df_pandas', 'sample_unsorted_df_dask', 'sample_unsorted_df_koalas'])
def sample_unsorted_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_unsorted_df_pandas():
    return pd.DataFrame({
        'id': [3, 1, 2, 0],
        'full_name': ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown', 'Ms. Paige Turner'],
        'email': ['john.smith@example.com', np.nan, 'team@featuretools.com', 'junk@example.com'],
        'phone_number': ['5555555555', '555-555-5555', '1-(555)-555-5555', '555-555-5555'],
        'age': [33, 25, 33, 57],
        'signup_date': pd.to_datetime(['2020-09-01', '2020-08-01', '2020-08-02', '2020-09-01']),
        'is_registered': [True, False, True, True],
    })


@pytest.fixture()
def sample_unsorted_df_dask(sample_unsorted_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason='Dask not installed, skipping')
    return dd.from_pandas(sample_unsorted_df_pandas, npartitions=2)


@pytest.fixture()
def sample_unsorted_df_koalas(sample_unsorted_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(sample_unsorted_df_pandas)


@pytest.fixture(params=['sample_series_pandas', 'sample_series_dask', 'sample_series_koalas'])
def sample_series(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_series_pandas():
    return pd.Series(['a', 'b', 'c'] + 10 * ['a', 'a', 'a'], name='sample_series').astype('category')


@pytest.fixture()
def sample_series_dask(sample_series_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(sample_series_pandas, npartitions=2)


@pytest.fixture()
def sample_series_koalas(sample_series_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(sample_series_pandas.astype('string'))


@pytest.fixture(params=['sample_datetime_series_pandas', 'sample_datetime_series_dask', 'sample_datetime_series_koalas'])
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
def sample_datetime_series_koalas(sample_datetime_series_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(sample_datetime_series_pandas)


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


@pytest.fixture()
def time_index_df_koalas(time_index_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(time_index_df_pandas)


@pytest.fixture(params=['time_index_df_pandas', 'time_index_df_dask', 'time_index_df_koalas'])
def time_index_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def numeric_time_index_df_pandas():
    return pd.DataFrame({
        'floats': pd.Series([1, 2, 3, 4], dtype='float'),
        'ints': pd.Series([1, -2, 3, 4], dtype='int64'),
        'with_null': pd.Series([1, pd.NA, 3, 4], dtype='Int64'),
    })


@pytest.fixture()
def numeric_time_index_df_dask(numeric_time_index_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(numeric_time_index_df_pandas, npartitions=2)


@pytest.fixture()
def numeric_time_index_df_koalas(numeric_time_index_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(numeric_time_index_df_pandas)


@pytest.fixture(params=['numeric_time_index_df_pandas', 'numeric_time_index_df_dask', 'numeric_time_index_df_koalas'])
def numeric_time_index_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def describe_df_pandas():
    index_data = [0, 1, 2, 3, 4, 5, 6, 7]
    boolean_data = [True, False, True, True, False, True, False, None]
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
    latlong_data = [(0, 0),
                    (1, 1),
                    (2, 2),
                    (3, 3),
                    (0, 0),
                    (np.nan, np.nan),
                    (np.nan, 6),
                    np.nan]
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
        'latlong_col': latlong_data,
    })


@pytest.fixture()
def describe_df_dask(describe_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(describe_df_pandas, npartitions=2)


@pytest.fixture()
def describe_df_koalas(describe_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(describe_df_pandas
                          .applymap(lambda tup: [None if pd.isnull(elt) else elt for elt in tup] if isinstance(tup, tuple) else tup)
                          .drop(columns='timedelta_col'))


@pytest.fixture(params=['describe_df_pandas', 'describe_df_dask', 'describe_df_koalas'])
def describe_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def df_same_mi_pandas():
    return pd.DataFrame({
        'ints': pd.Series([2, pd.NA, 5, 2], dtype='Int64'),
        'floats': pd.Series([1, None, 100, 1]),
        'nans': pd.Series([None, None, None, None]),
        'nat_lang': pd.Series(['this is a very long sentence inferred as a string', None, 'test', 'test']),
    })


@pytest.fixture()
def df_same_mi_dask(df_same_mi_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(df_same_mi_pandas, npartitions=2)


@pytest.fixture()
def df_same_mi_koalas(df_same_mi_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    df_same_mi_pandas['ints'] = df_same_mi_pandas['ints'].astype('float')
    df_same_mi_pandas['nans'] = df_same_mi_pandas['nans'].astype('float')
    return ks.DataFrame(df_same_mi_pandas)


@pytest.fixture(params=['df_same_mi_pandas', 'df_same_mi_dask', 'df_same_mi_koalas'])
def df_same_mi(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def df_mi_pandas():
    df = pd.DataFrame({
        'ints': pd.Series([1, 2, 1]),
        'bools': pd.Series([True, False, True]),
        'strs2': pd.Series(['bye', 'hi', 'bye']),
        'strs': pd.Series(['hi', 'hi', 'hi']),
        'dates': pd.Series(['2020-01-01', '2020-01-01', '1997-01-04'])
    })
    df = df.loc[df.index.repeat(4)].reset_index(drop=True)
    return df


@pytest.fixture()
def df_mi_dask(df_mi_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(df_mi_pandas, npartitions=1)


@pytest.fixture()
def df_mi_koalas(df_mi_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(df_mi_pandas)


@pytest.fixture(params=['df_mi_pandas', 'df_mi_dask', 'df_mi_koalas'])
def df_mi(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def df_mi_unique_pandas():
    df = pd.DataFrame({
        'unique': pd.Series(['hi', 'bye', 'hello', 'goodbye']),
        'unique_with_one_nan': pd.Series(['hi', 'bye', None, 'goodbye']),
        'unique_with_nans': pd.Series([1, None, None, 2]),
        'ints': pd.Series([1, 2, 1, 2]),
    })
    df = df.loc[df.index.repeat(5)].reset_index(drop=True)
    return df


@pytest.fixture()
def df_mi_unique_dask(df_mi_unique_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(df_mi_unique_pandas, npartitions=1)


@pytest.fixture()
def df_mi_unique_koalas(df_mi_unique_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(df_mi_unique_pandas)


@pytest.fixture(params=['df_mi_unique_pandas', 'df_mi_unique_dask', 'df_mi_unique_koalas'])
def df_mi_unique(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def categorical_df_pandas():
    return pd.DataFrame({
        'ints': pd.Series([1, 2, 3, 2]),
        'categories1': pd.Series([1, 100, 1, 100, 200, 200, 200, 200, 3, 100]),
        'bools': pd.Series([True, False, True, False]),
        'categories2': pd.Series(['test', 'test', 'test2', 'test']),
        'categories3': pd.Series(['test', 'test', 'test', np.nan]),
    })


@pytest.fixture()
def categorical_df_dask(categorical_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(categorical_df_pandas, npartitions=2)


@pytest.fixture()
def categorical_df_koalas(categorical_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(categorical_df_pandas)


@pytest.fixture(params=['categorical_df_pandas', 'categorical_df_dask', 'categorical_df_koalas'])
def categorical_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def empty_df_pandas():
    return pd.DataFrame({})


@pytest.fixture()
def empty_df_dask(empty_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(empty_df_pandas, npartitions=2)


# Cannot have an empty Koalas DataFrame
@pytest.fixture(params=['empty_df_pandas', 'empty_df_dask'])
def empty_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def small_df_pandas():
    return pd.DataFrame(pd.Series([pd.to_datetime('2020-09-01')] * 4, name='sample_datetime_series').astype('object'))


@pytest.fixture()
def small_df_dask(small_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(small_df_pandas, npartitions=1)


@pytest.fixture()
def small_df_koalas(small_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(small_df_pandas)


@pytest.fixture(params=['small_df_pandas', 'small_df_dask', 'small_df_koalas'])
def small_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def latlong_df_pandas():
    return pd.DataFrame({
        'tuple_ints': pd.Series([(1, 2), (3, 4)]),
        'tuple_strings': pd.Series([('1', '2'), ('3', '4')]),
        'string_tuple': pd.Series(['(1, 2)', '(3, 4)']),
        'bracketless_string_tuple': pd.Series(['1, 2', '3, 4']),
        'list_strings': pd.Series([['1', '2'], ['3', '4']]),
        'combo_tuple_types': pd.Series(['[1, 2]', '(3, 4)']),
        'null_value': pd.Series([np.nan, (3, 4)]),
        'null_latitude': pd.Series([(np.nan, 2.0), (3.0, 4.0)]),
        'both_null': pd.Series([(np.nan, np.nan), (3.0, 4.0)]),
    })


@pytest.fixture()
def latlong_df_dask(latlong_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(latlong_df_pandas, npartitions=2)


@pytest.fixture()
def latlong_df_koalas(latlong_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(latlong_df_pandas.applymap(lambda tup: list(tup) if isinstance(tup, tuple) else tup))


@pytest.fixture(params=['latlong_df_pandas', 'latlong_df_dask', 'latlong_df_koalas'])
def latlong_df(request):
    return request.getfixturevalue(request.param)


# LatLong Fixtures for testing access to latlong values
@pytest.fixture
def pandas_latlongs():
    return [
        pd.Series([(1.0, 2.0), (3.0, 4.0)]),
        pd.Series([('1', '2'), ('3', '4')]),
        pd.Series([['1', '2'], ['3', '4']]),
        pd.Series([(1, 2), (3, 4)]),
        pd.Series([[1, 2], [3, 4]]),
        pd.Series(['(1, 2)', '(3, 4)']),
        pd.Series(['1, 2', '3, 4']),
        pd.Series(['[1, 2]', '[3, 4]'])
    ]


@pytest.fixture
def dask_latlongs(pandas_latlongs):
    return [pd_to_dask(series) for series in pandas_latlongs]


@pytest.fixture
def koalas_latlongs(pandas_latlongs):
    return [pd_to_koalas(series.apply(lambda tup: list(tup) if isinstance(tup, tuple) else tup)) for series in pandas_latlongs]


@pytest.fixture(params=['pandas_latlongs', 'dask_latlongs', 'koalas_latlongs'])
def latlongs(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def whitespace_df_pandas():
    return pd.DataFrame({
        'id': [0, 1, 2, 3, 4, 5],
        'comments': [
            '\nleading newline',
            'trailing newline\n',
            '\n',
            'newline in \n the middle',
            '    leading whitespace',
            'trailing whitespace ',
        ]
    })


@pytest.fixture()
def whitespace_df_dask(whitespace_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(whitespace_df_pandas, npartitions=2)


@pytest.fixture()
def whitespace_df_koalas(whitespace_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(whitespace_df_pandas)


@pytest.fixture(params=['whitespace_df_pandas', 'whitespace_df_dask', 'whitespace_df_koalas'])
def whitespace_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def falsy_names_df_pandas():
    return pd.DataFrame({
        0: ['a', 'b', 'c'],
        '': [1, 2, 3],
    })


@pytest.fixture()
def falsy_names_df_dask(falsy_names_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(falsy_names_df_pandas, npartitions=2)


@pytest.fixture()
def falsy_names_df_koalas(falsy_names_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(falsy_names_df_pandas.applymap(lambda tup: list(tup) if isinstance(tup, tuple) else tup))


@pytest.fixture(params=['falsy_names_df_pandas', 'falsy_names_df_dask', 'falsy_names_df_koalas'])
def falsy_names_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_column_names(sample_df_pandas):
    return sample_df_pandas.columns.to_list()


@pytest.fixture()
def sample_inferred_logical_types():
    return {'id': Integer,
            'full_name': Unknown,
            'email': EmailAddress,
            'phone_number': Unknown,
            'age': IntegerNullable,
            'signup_date': Datetime,
            'is_registered': BooleanNullable,
            'double': Double,
            'double_with_nan': Double,
            'integer': Integer,
            'nullable_integer': IntegerNullable,
            'boolean': Boolean,
            'categorical': Categorical,
            'datetime_with_NaT': Datetime,
            }


@pytest.fixture()
def sample_correct_logical_types():
    return {'id': Integer,
            'full_name': PersonFullName,
            'email': EmailAddress,
            'phone_number': PhoneNumber,
            'age': IntegerNullable,
            'signup_date': Datetime,
            'is_registered': BooleanNullable,
            'double': Double,
            'double_with_nan': Double,
            'integer': Integer,
            'nullable_integer': IntegerNullable,
            'boolean': Boolean,
            'categorical': Categorical,
            'datetime_with_NaT': Datetime,
            }


@pytest.fixture()
def serialize_df_pandas():
    df = pd.DataFrame({
        'id': [0, 1, 2],
        'cat_int': [1, 2, 1],
        'ord_int': [1, 2, 1],
        'cat_float': [1.0, 2.0, 1.0],
        'ord_float': [1.0, 2.0, 1.0],
        'cat_bool': [True, False, True],
        'ord_bool': [True, False, True],
    })
    return df


@pytest.fixture()
def serialize_df_dask(serialize_df_pandas):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(serialize_df_pandas, npartitions=2)


@pytest.fixture()
def serialize_df_koalas(serialize_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(serialize_df_pandas)


@pytest.fixture(params=['serialize_df_pandas', 'serialize_df_dask', 'serialize_df_koalas'])
def serialize_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def pandas_datetimes():
    return [
        pd.Series(['3/11/2000', '3/12/2000', '3/13/2000', '3/14/2000']),
        pd.Series(['3/11/2000', np.nan, '3/13/2000', '3/14/2000']),
    ]


@pytest.fixture()
def dask_datetimes(pandas_datetimes):
    return [pd_to_dask(series) for series in pandas_datetimes]


@pytest.fixture()
def koalas_datetimes(pandas_datetimes):
    return [pd_to_koalas(series) for series in pandas_datetimes]


@pytest.fixture(params=['pandas_datetimes', 'dask_datetimes', 'koalas_datetimes'])
def datetimes(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def outliers_df_pandas():
    return pd.DataFrame({
        'has_outliers': [93, 42, 37, -16, 49, 42, 36, 57, 60, 23],
        'no_outliers': [60, 42, 37, 23, 49, 42, 36, 57, 60, 23.0],
        'non_numeric': ['a'] * 10,
        'has_outliers_with_nans': [None, 42, 37, -16, 49, 93, 36, 57, 60, 23],
        'nans': pd.Series([None] * 10, dtype='float64'),
    })


@pytest.fixture()
def outliers_df_dask(outliers_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason='Dask not installed, skipping')
    return dd.from_pandas(outliers_df_pandas, npartitions=2)


@pytest.fixture()
def outliers_df_koalas(outliers_df_pandas):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(outliers_df_pandas)


@pytest.fixture(params=['outliers_df_pandas', 'outliers_df_dask', 'outliers_df_koalas'])
def outliers_df(request):
    return request.getfixturevalue(request.param)
