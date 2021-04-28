import numpy as np
import pandas as pd

from woodwork.logical_types import (
    URL,
    Age,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    EmailAddress,
    Filepath,
    Integer,
    IntegerNullable,
    IPAddress,
    LatLong,
    NaturalLanguage,
    Ordinal,
    PersonFullName,
    PhoneNumber,
    PostalCode,
    SubRegionCode,
    Timedelta
)
from woodwork.statistics_utils import (
    _get_describe_dict,
    _get_mode,
    _make_categorical_for_mutual_info,
    _replace_nans_for_mutual_info
)
from woodwork.tests.testing_utils import mi_between_cols, to_pandas
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_get_mode():
    series_list = [
        pd.Series([1, 2, 3, 4, 2, 2, 3]),
        pd.Series(['a', 'b', 'b', 'c', 'b']),
        pd.Series([3, 2, 3, 2]),
        pd.Series([np.nan, np.nan, np.nan]),
        pd.Series([pd.NA, pd.NA, pd.NA]),
        pd.Series([1, 2, np.nan, 2, np.nan, 3, 2]),
        pd.Series([1, 2, pd.NA, 2, pd.NA, 3, 2])
    ]

    answer_list = [2, 'b', 2, None, None, 2, 2]

    for series, answer in zip(series_list, answer_list):
        mode = _get_mode(series)
        if answer is None:
            assert mode is None
        else:
            assert mode == answer


def test_accessor_replace_nans_for_mutual_info():
    df_nans = pd.DataFrame({
        'ints': pd.Series([2, pd.NA, 5, 2], dtype='Int64'),
        'floats': pd.Series([3.3, None, 2.3, 1.3]),
        'bools': pd.Series([True, None, True, False]),
        'bools_pdna': pd.Series([True, pd.NA, True, False], dtype='boolean'),
        'int_to_cat_nan': pd.Series([1, np.nan, 3, 1], dtype='category'),
        'str': pd.Series(['test', np.nan, 'test2', 'test']),
        'str_no_nan': pd.Series(['test', 'test2', 'test2', 'test']),
        'dates': pd.Series(['2020-01-01', None, '2020-01-02', '2020-01-03'])
    })
    df_nans.ww.init()
    formatted_df = _replace_nans_for_mutual_info(df_nans.ww.schema, df_nans.copy())

    assert isinstance(formatted_df, pd.DataFrame)

    assert formatted_df['ints'].equals(pd.Series([2, 3, 5, 2], dtype='Int64'))
    assert formatted_df['floats'].equals(pd.Series([3.3, 2.3, 2.3, 1.3], dtype='float'))
    assert formatted_df['bools'].equals(pd.Series([True, True, True, False], dtype='category'))
    assert formatted_df['bools_pdna'].equals(pd.Series([True, True, True, False], dtype='boolean'))
    assert formatted_df['int_to_cat_nan'].equals(pd.Series([1, 1, 3, 1], dtype='category'))
    assert formatted_df['str'].equals(pd.Series(['test', 'test', 'test2', 'test'], dtype='category'))
    assert formatted_df['str_no_nan'].equals(pd.Series(['test', 'test2', 'test2', 'test'], dtype='category'))
    assert formatted_df['dates'].equals(pd.Series(['2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03'], dtype='datetime64[ns]'))


def test_accessor_make_categorical_for_mutual_info():
    df = pd.DataFrame({
        'ints1': pd.Series([1, 2, 3, 2]),
        'ints2': pd.Series([1, 100, 1, 100]),
        'ints3': pd.Series([1, 2, 3, 2], dtype='Int64'),
        'bools': pd.Series([True, False, True, False]),
        'booleans': pd.Series([True, False, True, False], dtype='boolean'),
        'categories': pd.Series(['test', 'test2', 'test2', 'test']),
        'dates': pd.Series(['2020-01-01', '2019-01-02', '2020-08-03', '1997-01-04'])
    })
    df.ww.init()
    formatted_num_bins_df = _make_categorical_for_mutual_info(df.ww.schema, df.copy(), num_bins=4)

    assert isinstance(formatted_num_bins_df, pd.DataFrame)

    assert formatted_num_bins_df['ints1'].equals(pd.Series([0, 1, 3, 1], dtype='int8'))
    assert formatted_num_bins_df['ints2'].equals(pd.Series([0, 1, 0, 1], dtype='int8'))
    assert formatted_num_bins_df['ints3'].equals(pd.Series([0, 1, 3, 1], dtype='int8'))
    assert formatted_num_bins_df['bools'].equals(pd.Series([1, 0, 1, 0], dtype='int8'))
    assert formatted_num_bins_df['booleans'].equals(pd.Series([1, 0, 1, 0], dtype='int8'))
    assert formatted_num_bins_df['categories'].equals(pd.Series([0, 1, 1, 0], dtype='int8'))
    assert formatted_num_bins_df['dates'].equals(pd.Series([2, 1, 3, 0], dtype='int8'))


def test_mutual_info_same(df_same_mi):
    df_same_mi.ww.init()

    mi = df_same_mi.ww.mutual_information()

    cols_used = set(np.unique(mi[['column_1', 'column_2']].values))
    assert 'nans' not in cols_used
    assert 'nat_lang' not in cols_used
    assert mi.shape[0] == 1
    assert mi_between_cols('floats', 'ints', mi) == 1.0


def test_mutual_info(df_mi):
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
    assert mi.shape[0] == 10
    assert (mi['mutual_info'] == 1.0).all()

    mi = df_mi.ww.mutual_information(num_bins=2)
    assert mi.shape[0] == 10
    np.testing.assert_almost_equal(mi_between_cols('bools', 'ints', mi), 0.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('strs', 'ints', mi), 1.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('bools', 'strs', mi), 0, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'strs', mi), 1.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'ints', mi), 1.0, 3)

    # Confirm that none of this changed the underlying df
    pd.testing.assert_frame_equal(to_pandas(df_mi), to_pandas(original_df))


def test_mutual_info_on_index(sample_df):
    sample_df.ww.init(index='id')
    mi = sample_df.ww.mutual_information()

    assert not ('id' in mi['column_1'].values or 'id' in mi['column_2'].values)

    mi = sample_df.ww.mutual_information(include_index=True)
    assert 'id' in mi['column_1'].values or 'id' in mi['column_2'].values


def test_mutual_info_returns_empty_df_properly(sample_df):
    schema_df = sample_df[['id', 'age']]
    schema_df.ww.init(index='id')

    mi = schema_df.ww.mutual_information()
    assert mi.empty


def test_mutual_info_sort(df_mi):
    df_mi.ww.init()
    mi = df_mi.ww.mutual_information()

    for i in range(len(mi['mutual_info']) - 1):
        assert mi['mutual_info'].iloc[i] >= mi['mutual_info'].iloc[i + 1]


def test_mutual_info_dict(df_mi):
    df_mi.ww.init()
    mi_dict = df_mi.ww.mutual_information_dict()
    mi = df_mi.ww.mutual_information()

    pd.testing.assert_frame_equal(pd.DataFrame(mi_dict), mi)


def test_mutual_info_unique_cols(df_mi_unique):
    df_mi_unique.ww.init()
    mi = df_mi_unique.ww.mutual_information()

    cols_used = set(np.unique(mi[['column_1', 'column_2']].values))
    assert 'unique' in cols_used
    assert 'unique_with_one_nan' in cols_used
    assert 'unique_with_nans' in cols_used
    assert 'ints' in cols_used


def test_get_describe_dict(describe_df):
    describe_df.ww.init(index='index_col')

    stats_dict = _get_describe_dict(describe_df)
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
    describe_df.ww.init(index='index_col')
    stats_df = describe_df.ww.describe()
    assert 'index_col' not in stats_df.columns


def test_describe_accessor_method(describe_df):
    categorical_ltypes = [Categorical,
                          CountryCode,
                          Ordinal(order=('yellow', 'red', 'blue')),
                          PostalCode,
                          SubRegionCode]
    boolean_ltypes = [BooleanNullable]
    non_nullable_boolean_ltypes = [Boolean]
    datetime_ltypes = [Datetime]
    formatted_datetime_ltypes = [Datetime(datetime_format='%Y~%m~%d')]
    timedelta_ltypes = [Timedelta]
    nullable_numeric_ltypes = [Double, IntegerNullable, AgeNullable]
    non_nullable_numeric_ltypes = [Integer, Age]
    natural_language_ltypes = [EmailAddress, Filepath, PersonFullName, IPAddress,
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
        expected_dtype = 'string'
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
        assert expected_vals.equals(stats_df['category_col'].dropna())

    # Test nullable boolean columns
    boolean_data = describe_df[['boolean_col']]
    for ltype in boolean_ltypes:
        expected_dtype = ltype.primary_dtype
        expected_vals = pd.Series({
            'physical_type': expected_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 7,
            'nan_count': 1,
            'mode': True,
            'num_true': 4,
            'num_false': 3}, name='boolean_col')
        boolean_data.ww.init(logical_types={'boolean_col': ltype}, semantic_tags={'boolean_col': 'custom_tag'})
        stats_df = boolean_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'boolean_col'}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df['boolean_col'].dropna())

    # Test non-nullable boolean columns
    boolean_data = describe_df[['boolean_col']].fillna(True)
    for ltype in non_nullable_boolean_ltypes:
        expected_dtype = ltype.primary_dtype
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
        assert expected_vals.equals(stats_df['boolean_col'].dropna())

    # Test datetime columns
    datetime_data = describe_df[['datetime_col']]
    for ltype in datetime_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.primary_dtype,
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
        assert expected_vals.equals(stats_df['datetime_col'].dropna())

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
            'physical_type': ltype.primary_dtype,
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
        assert expected_vals.equals(stats_df['formatted_datetime_col'].dropna())

    # Test timedelta columns - Skip for Koalas
    if not (ks and isinstance(describe_df, ks.DataFrame)):
        timedelta_data = describe_df['timedelta_col']
        for ltype in timedelta_ltypes:
            expected_vals = pd.Series({
                'physical_type': ltype.primary_dtype,
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
            assert expected_vals.equals(stats_df['col'].dropna())

    # Test numeric columns with nullable ltypes
    numeric_data = describe_df[['numeric_col']]
    for ltype in nullable_numeric_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.primary_dtype,
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
        assert expected_vals.equals(stats_df['numeric_col'].dropna())

    # Test numeric with non-nullable ltypes
    numeric_data = describe_df[['numeric_col']].fillna(0)
    for ltype in non_nullable_numeric_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.primary_dtype,
            'logical_type': ltype,
            'semantic_tags': {'numeric', 'custom_tag'},
            'count': 8,
            'nunique': 7,
            'nan_count': 0,
            'mean': 18.25,
            'mode': 10,
            'std': 18.460382289804137,
            'min': 0,
            'first_quartile': 7.75,
            'second_quartile': 13.5,
            'third_quartile': 23,
            'max': 56}, name='numeric_col')
        numeric_data.ww.init(logical_types={'numeric_col': ltype}, semantic_tags={'numeric_col': 'custom_tag'})
        stats_df = numeric_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'numeric_col'}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df['numeric_col'].dropna())

    # Test natural language columns
    natural_language_data = describe_df[['natural_language_col']]
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
        assert expected_vals.equals(stats_df['natural_language_col'].dropna())

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
        assert expected_vals.equals(stats_df['latlong_col'].dropna())


def test_describe_with_improper_tags(describe_df):
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


def test_describe_with_no_semantic_tags(describe_df):
    df = describe_df.copy()[['category_col', 'numeric_col']]

    logical_types = {
        'category_col': Categorical,
        'numeric_col': IntegerNullable,
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


def test_describe_with_include(sample_df):
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

    logical_types_df = sample_df.ww.describe([Datetime, BooleanNullable])
    assert 'signup_date', 'is_registered' in logical_types_df.columns
    assert len(logical_types_df.columns) == 2

    multi_params_df = sample_df.ww.describe(['age', 'tag1', Datetime])
    expected = ['full_name', 'age', 'signup_date']
    for col_name in expected:
        assert col_name in multi_params_df.columns
    multi_params_df['full_name'].equals(col_name_df['full_name'])
    multi_params_df['full_name'].equals(sample_df.ww.describe()['full_name'])


def test_describe_with_no_match(sample_df):
    sample_df.ww.init()
    df = sample_df.ww.describe(include=['wrongname'])
    assert df.empty


def test_value_counts(categorical_df):
    logical_types = {
        'ints': IntegerNullable,
        'categories1': Categorical,
        'bools': Boolean,
        'categories2': Categorical,
        'categories3': Categorical,
    }
    categorical_df.ww.init(logical_types=logical_types)
    val_cts = categorical_df.ww.value_counts()
    for col in categorical_df.ww.columns:
        if col in ['ints', 'bools']:
            assert col not in val_cts
        else:
            assert col in val_cts

    expected_cat1 = [{'value': 200, 'count': 4}, {'value': 100, 'count': 3}, {'value': 1, 'count': 2}, {'value': 3, 'count': 1}]
    # Koalas converts numeric categories to strings, so we need to update the expected values for this
    # Koalas will result in `None` instead of `np.nan` in categorical columns
    if ks and isinstance(categorical_df, ks.DataFrame):
        updated_results = []
        for items in expected_cat1:
            updated_results.append({k: (str(v) if k == 'value' else v) for k, v in items.items()})
        expected_cat1 = updated_results

    assert val_cts['categories1'] == expected_cat1
    assert val_cts['categories2'] == [{'value': np.nan, 'count': 6}, {'value': 'test', 'count': 3}, {'value': 'test2', 'count': 1}]
    assert val_cts['categories3'] == [{'value': np.nan, 'count': 7}, {'value': 'test', 'count': 3}]

    val_cts_descending = categorical_df.ww.value_counts(ascending=True)
    for col, vals in val_cts_descending.items():
        for i in range(len(vals)):
            assert vals[i]['count'] == val_cts[col][-i - 1]['count']

    val_cts_dropna = categorical_df.ww.value_counts(dropna=True)
    assert val_cts_dropna['categories3'] == [{'value': 'test', 'count': 3}]

    val_cts_2 = categorical_df.ww.value_counts(top_n=2)
    for col in val_cts_2:
        assert len(val_cts_2[col]) == 2
