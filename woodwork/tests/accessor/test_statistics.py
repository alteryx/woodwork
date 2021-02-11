import numpy as np
import pandas as pd
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
    LogicalType,
    NaturalLanguage,
    Ordinal,
    PhoneNumber,
    SubRegionCode,
    Timedelta,
    ZIPCode
)
from woodwork.tests.testing_utils import (
    mi_between_cols,
    to_pandas,
    xfail_dask_and_koalas
)
from woodwork.statistics_utils import _get_describe_dict
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_accessor_replace_nans_for_mutual_info():
    df_nans = pd.DataFrame({
        'ints': pd.Series([2, pd.NA, 5, 2], dtype='Int64'),
        'floats': pd.Series([3.3, None, 2.3, 1.3]),
        'bools': pd.Series([True, None, True, False]),
        'int_to_cat_nan': pd.Series([1, np.nan, 3, 1], dtype='category'),
        'str': pd.Series(['test', np.nan, 'test2', 'test']),
        'str_no_nan': pd.Series(['test', 'test2', 'test2', 'test']),
        'dates': pd.Series(['2020-01-01', None, '2020-01-02', '2020-01-03'])
    })
    df_nans.ww.init()
    formatted_df = df_nans.ww._replace_nans_for_mutual_info(df_nans.copy())

    assert isinstance(formatted_df, pd.DataFrame)

    assert formatted_df['ints'].equals(pd.Series([2, 3, 5, 2], dtype='Int64'))
    assert formatted_df['floats'].equals(pd.Series([3.3, 2.3, 2.3, 1.3], dtype='float'))
    assert formatted_df['bools'].equals(pd.Series([True, True, True, False], dtype='category'))
    assert formatted_df['int_to_cat_nan'].equals(pd.Series([1, 1, 3, 1], dtype='category'))
    assert formatted_df['str'].equals(pd.Series(['test', 'test', 'test2', 'test'], dtype='category'))
    assert formatted_df['str_no_nan'].equals(pd.Series(['test', 'test2', 'test2', 'test'], dtype='category'))
    assert formatted_df['dates'].equals(pd.Series(['2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03'], dtype='datetime64[ns]'))


def test_accessor_make_categorical_for_mutual_info():
    df = pd.DataFrame({
        'ints1': pd.Series([1, 2, 3, 2]),
        'ints2': pd.Series([1, 100, 1, 100]),
        'bools': pd.Series([True, False, True, False]),
        'categories': pd.Series(['test', 'test2', 'test2', 'test']),
        'dates': pd.Series(['2020-01-01', '2019-01-02', '2020-08-03', '1997-01-04'])
    })
    df.ww.init()
    formatted_num_bins_df = df.ww._make_categorical_for_mutual_info(df.copy(), num_bins=4)

    assert isinstance(formatted_num_bins_df, pd.DataFrame)

    assert formatted_num_bins_df['ints1'].equals(pd.Series([0, 1, 3, 1], dtype='int8'))
    assert formatted_num_bins_df['ints2'].equals(pd.Series([0, 1, 0, 1], dtype='int8'))
    assert formatted_num_bins_df['bools'].equals(pd.Series([1, 0, 1, 0], dtype='int8'))
    assert formatted_num_bins_df['categories'].equals(pd.Series([0, 1, 1, 0], dtype='int8'))
    assert formatted_num_bins_df['dates'].equals(pd.Series([2, 1, 3, 0], dtype='int8'))


def test_mutual_info_same(df_same_mi):
    xfail_dask_and_koalas(df_same_mi)

    df_same_mi.ww.init()

    mi = df_same_mi.ww.mutual_information()

    cols_used = set(np.unique(mi[['column_1', 'column_2']].values))
    assert 'nans' not in cols_used
    assert 'nat_lang' not in cols_used
    assert mi.shape[0] == 1
    assert mi_between_cols('floats', 'ints', mi) == 1.0


def test_mutual_info(df_mi):
    xfail_dask_and_koalas(df_mi)

    df_mi.ww.init(logical_types={'dates': Datetime(datetime_format='%Y-%m-%d')})
    original_df = df_mi.copy()
    mi = df_mi.ww.mutual_information()
    assert mi.shape[0] == 10

    np.testing.assert_almost_equal(mi_between_cols('ints', 'bools', mi), 1.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('ints', 'strs', mi), 0.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('strs', 'bools', mi), 0, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'ints', mi), 0.274, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'bools', mi), 0.274, 3)

    mi_many_rows = df_mi.ww.mutual_information(nrows=100000)
    pd.testing.assert_frame_equal(mi, mi_many_rows)

    mi = df_mi.ww.mutual_information(nrows=1)
    assert mi.shape[0] == 0

    mi = df_mi.ww.mutual_information(num_bins=2)
    assert mi.shape[0] == 10
    np.testing.assert_almost_equal(mi_between_cols('bools', 'ints', mi), 0.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('strs', 'ints', mi), 1.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('bools', 'strs', mi), 0, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'strs', mi), 1.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'ints', mi), 1.0, 3)

    # Confirm that none of this changed the underlying df
    pd.testing.assert_frame_equal(to_pandas(df_mi), to_pandas(original_df))


def test_mutual_info_does_not_include_index(sample_df):
    xfail_dask_and_koalas(sample_df)

    sample_df.ww.init(index='id')
    mi = sample_df.ww.mutual_information()

    assert 'id' not in mi['column_1'].values


def test_mutual_info_returns_empty_df_properly(sample_df):
    xfail_dask_and_koalas(sample_df)

    schema_df = sample_df[['id', 'age']]
    schema_df.ww.init(index='id')

    mi = schema_df.ww.mutual_information()
    assert mi.empty


def test_mutual_info_sort(df_mi):
    xfail_dask_and_koalas(df_mi)

    df_mi.ww.init()
    mi = df_mi.ww.mutual_information()

    for i in range(len(mi['mutual_info']) - 1):
        assert mi['mutual_info'].iloc[i] >= mi['mutual_info'].iloc[i + 1]


def test_mutual_info_dict(df_mi):
    xfail_dask_and_koalas(df_mi)

    df_mi.ww.init()
    mi_dict = df_mi.ww.mutual_information_dict()
    mi = df_mi.ww.mutual_information()

    pd.testing.assert_frame_equal(pd.DataFrame(mi_dict), mi)


def test_mutual_info_unique_cols(df_mi_unique):
    xfail_dask_and_koalas(df_mi_unique)

    df_mi_unique.ww.init()
    mi = df_mi_unique.ww.mutual_information()

    cols_used = set(np.unique(mi[['column_1', 'column_2']].values))
    assert 'unique' not in cols_used
    assert 'unique_with_one_nan' not in cols_used
    assert 'unique_with_nans' in cols_used
    assert 'ints' in cols_used


def test_describe_dict(describe_df):
    xfail_dask_and_koalas(describe_df)

    describe_df.ww.init(index='index_col')

    stats_dict = _get_describe_dict(describe_df, describe_df.ww.schema)
    index_order = ['physical_type',
                   'logical_type',
                   'semantic_tags',
                   'count',
                   'nunique',
                   'nan_count',
                   'mean',
                   'mode',
                   'std',
                   'min',
                   'first_quartile',
                   'second_quartile',
                   'third_quartile',
                   'max',
                   'num_true',
                   'num_false']
    stats_dict_to_df = pd.DataFrame(stats_dict).reindex(index_order)
    stats_df = describe_df.ww.describe()
    pd.testing.assert_frame_equal(stats_df, stats_dict_to_df)


def test_describe_does_not_include_index(describe_df):
    xfail_dask_and_koalas(describe_df)

    describe_df.ww.init(index='index_col')
    stats_df = describe_df.ww.describe()
    assert 'index_col' not in stats_df.columns


def test_describe_accessor_method(describe_df):
    xfail_dask_and_koalas(describe_df)

    categorical_ltypes = [Categorical,
                          CountryCode,
                          Ordinal(order=('yellow', 'red', 'blue')),
                          SubRegionCode,
                          ZIPCode]
    boolean_ltypes = [Boolean]
    datetime_ltypes = [Datetime]
    formatted_datetime_ltypes = [Datetime(datetime_format='%Y~%m~%d')]
    timedelta_ltypes = [Timedelta]
    numeric_ltypes = [Double, Integer]
    natural_language_ltypes = [EmailAddress, Filepath, FullName, IPAddress,
                               PhoneNumber, URL]
    latlong_ltypes = [LatLong]

    expected_index = ['physical_type',
                      'logical_type',
                      'semantic_tags',
                      'count',
                      'nunique',
                      'nan_count',
                      'mean',
                      'mode',
                      'std',
                      'min',
                      'first_quartile',
                      'second_quartile',
                      'third_quartile',
                      'max',
                      'num_true',
                      'num_false']

    # Test categorical columns
    category_data = describe_df[['category_col']]
    if ks and isinstance(category_data, ks.DataFrame):
        expected_dtype = 'object'
    else:
        expected_dtype = 'category'

    for ltype in categorical_ltypes:
        expected_vals = pd.Series({
            'physical_type': expected_dtype,
            'logical_type': ltype,
            'semantic_tags': {'category', 'custom_tag'},
            'count': 7,
            'nunique': 3,
            'nan_count': 1,
            'mode': 'red'}, name='category_col')
        category_data.ww.init(logical_types={'category_col': ltype}, semantic_tags={'category_col': 'custom_tag'})
        stats_df = category_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'category_col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['category_col'].dropna())

    # Test boolean columns
    boolean_data = describe_df[['boolean_col']]
    if ks and isinstance(category_data, ks.DataFrame):
        expected_dtype = 'bool'
    else:
        expected_dtype = 'boolean'
    for ltype in boolean_ltypes:
        expected_vals = pd.Series({
            'physical_type': expected_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 8,
            'nan_count': 0,
            'mode': True,
            'num_true': 5,
            'num_false': 3}, name='boolean_col')
        boolean_data.ww.init(logical_types={'boolean_col': ltype}, semantic_tags={'boolean_col': 'custom_tag'})
        stats_df = boolean_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'boolean_col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['boolean_col'].dropna())

    # Test datetime columns
    datetime_data = describe_df[['datetime_col']]
    for ltype in datetime_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 7,
            'nunique': 6,
            'nan_count': 1,
            'mean': pd.Timestamp('2020-01-19 09:25:42.857142784'),
            'mode': pd.Timestamp('2020-02-01 00:00:00'),
            'min': pd.Timestamp('2020-01-01 00:00:00'),
            'max': pd.Timestamp('2020-02-02 18:00:00')}, name='datetime_col')
        datetime_data.ww.init(logical_types={'datetime_col': ltype}, semantic_tags={'datetime_col': 'custom_tag'})
        stats_df = datetime_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'datetime_col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['datetime_col'].dropna())

    # Test formatted datetime columns
    formatted_datetime_data = describe_df[['formatted_datetime_col']]
    for ltype in formatted_datetime_ltypes:
        converted_to_datetime = pd.to_datetime(['2020-01-01',
                                                '2020-02-01',
                                                '2020-03-01',
                                                '2020-02-02',
                                                '2020-03-02',
                                                pd.NaT,
                                                '2020-02-01',
                                                '2020-01-02'])
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 7,
            'nunique': 6,
            'nan_count': 1,
            'mean': converted_to_datetime.mean(),
            'mode': pd.to_datetime('2020-02-01'),
            'min': converted_to_datetime.min(),
            'max': converted_to_datetime.max()}, name='formatted_datetime_col')
        formatted_datetime_data.ww.init(
            logical_types={'formatted_datetime_col': ltype},
            semantic_tags={'formatted_datetime_col': 'custom_tag'})
        stats_df = formatted_datetime_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'formatted_datetime_col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['formatted_datetime_col'].dropna())

    # Test timedelta columns - Skip for Koalas
    if not (ks and isinstance(describe_df, ks.DataFrame)):
        timedelta_data = describe_df['timedelta_col']
        for ltype in timedelta_ltypes:
            expected_vals = pd.Series({
                'physical_type': ltype.pandas_dtype,
                'logical_type': ltype,
                'semantic_tags': {'custom_tag'},
                'count': 7,
                'nan_count': 1,
                'mode': pd.Timedelta('31days')}, name='col')
            df = pd.DataFrame({'col': timedelta_data})
            df.ww.init(logical_types={'col': ltype}, semantic_tags={'col': 'custom_tag'})
            stats_df = df.ww.describe()
            assert isinstance(stats_df, pd.DataFrame)
            assert set(stats_df.columns) == {'col'}
            assert stats_df.index.tolist() == expected_index
            pd.testing.assert_series_equal(expected_vals, stats_df['col'].dropna())

    # Test numeric columns
    numeric_data = describe_df[['numeric_col']]
    for ltype in numeric_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'numeric', 'custom_tag'},
            'count': 7,
            'nunique': 6,
            'nan_count': 1,
            'mean': 20.857142857142858,
            'mode': 10,
            'std': 18.27957486220227,
            'min': 1,
            'first_quartile': 10,
            'second_quartile': 17,
            'third_quartile': 26,
            'max': 56}, name='numeric_col')
        numeric_data.ww.init(logical_types={'numeric_col': ltype}, semantic_tags={'numeric_col': 'custom_tag'})
        stats_df = numeric_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'numeric_col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['numeric_col'].dropna(), check_exact=False)

    # Test natural language columns
    natural_language_data = describe_df[['natural_language_col']]
    if ks and isinstance(category_data, ks.DataFrame):
        expected_dtype = 'object'
    else:
        expected_dtype = 'string'
    for ltype in natural_language_ltypes:
        expected_vals = pd.Series({
            'physical_type': expected_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 7,
            'nan_count': 1,
            'mode': 'Duplicate sentence.'}, name='natural_language_col')
        natural_language_data.ww.init(
            logical_types={'natural_language_col': ltype},
            semantic_tags={'natural_language_col': 'custom_tag'})
        stats_df = natural_language_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'natural_language_col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['natural_language_col'].dropna())

    # Test latlong columns
    latlong_data = describe_df[['latlong_col']]
    expected_dtype = 'object'
    for ltype in latlong_ltypes:
        mode = [0, 0] if ks and isinstance(describe_df, ks.DataFrame) else (0, 0)
        expected_vals = pd.Series({
            'physical_type': expected_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 6,
            'nan_count': 2,
            'mode': mode}, name='latlong_col')
        latlong_data.ww.init(
            logical_types={'latlong_col': ltype},
            semantic_tags={'latlong_col': 'custom_tag'})
        stats_df = latlong_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'latlong_col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['latlong_col'].dropna())


def test_describe_with_improper_tags(describe_df):
    xfail_dask_and_koalas(describe_df)

    df = describe_df.copy()[['boolean_col', 'natural_language_col']]

    logical_types = {
        'boolean_col': Boolean,
        'natural_language_col': NaturalLanguage,
    }
    semantic_tags = {
        'boolean_col': 'category',
        'natural_language_col': 'numeric',
    }

    df.ww.init(logical_types=logical_types, semantic_tags=semantic_tags)
    stats_df = df.ww.describe()

    # Make sure boolean stats were computed with improper 'category' tag
    assert stats_df['boolean_col']['logical_type'] == Boolean
    assert stats_df['boolean_col']['semantic_tags'] == {'category'}
    # Make sure numeric stats were not computed with improper 'numeric' tag
    assert stats_df['natural_language_col']['semantic_tags'] == {'numeric'}
    assert stats_df['natural_language_col'][['mean', 'std', 'min', 'max']].isnull().all()


def test_datatable_describe_with_no_semantic_tags(describe_df):
    xfail_dask_and_koalas(describe_df)

    df = describe_df.copy()[['category_col', 'numeric_col']]

    logical_types = {
        'category_col': Categorical,
        'numeric_col': Integer,
    }

    df.ww.init(logical_types=logical_types, use_standard_tags=False)
    stats_df = df.ww.describe()
    assert df.ww.semantic_tags['category_col'] == set()
    assert df.ww.semantic_tags['numeric_col'] == set()

    # Make sure category stats were computed
    assert stats_df['category_col']['semantic_tags'] == set()
    assert stats_df['category_col']['nunique'] == 3
    # Make sure numeric stats were computed
    assert stats_df['numeric_col']['semantic_tags'] == set()
    np.testing.assert_almost_equal(stats_df['numeric_col']['mean'], 20.85714, 5)


def test_datatable_describe_with_include(sample_df):
    xfail_dask_and_koalas(sample_df)

    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'age': ['numeric', 'age']
    }
    sample_df.ww.init(semantic_tags=semantic_tags)

    col_name_df = sample_df.ww.describe(include=['full_name'])
    assert col_name_df.shape == (16, 1)
    assert 'full_name', 'email' in col_name_df.columns

    semantic_tags_df = sample_df.ww.describe(['tag1', 'tag2'])
    assert 'full_name' in col_name_df.columns
    assert len(semantic_tags_df.columns) == 2

    logical_types_df = sample_df.ww.describe([Datetime, Boolean])
    assert 'signup_date', 'is_registered' in logical_types_df.columns
    assert len(logical_types_df.columns) == 2

    multi_params_df = sample_df.ww.describe(['age', 'tag1', Datetime])
    expected = ['full_name', 'age', 'signup_date']
    for col_name in expected:
        assert col_name in multi_params_df.columns
    multi_params_df['full_name'].equals(col_name_df['full_name'])
    multi_params_df['full_name'].equals(sample_df.ww.describe()['full_name'])
