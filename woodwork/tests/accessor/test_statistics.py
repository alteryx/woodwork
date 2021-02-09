import numpy as np
import pandas as pd

from woodwork.logical_types import Datetime
from woodwork.tests.testing_utils import (
    mi_between_cols,
    to_pandas,
    xfail_dask_and_koalas
)


def test_accessor_handle_nans_for_mutual_info():
    df_nans = pd.DataFrame({
        'nans': pd.Series([None, None, None, None]),
        'ints': pd.Series([2, pd.NA, 5, 2], dtype='Int64'),
        'floats': pd.Series([3.3, None, 2.3, 1.3]),
        'bools': pd.Series([True, None, True, False]),
        'int_to_cat_nan': pd.Series([1, np.nan, 3, 1], dtype='category'),
        'str': pd.Series(['test', np.nan, 'test2', 'test']),
        'str_no_nan': pd.Series(['test', 'test2', 'test2', 'test']),
        'dates': pd.Series(['2020-01-01', None, '2020-01-02', '2020-01-03'])
    })
    df_nans.ww.init()
    formatted_df = df_nans.ww._handle_nans_for_mutual_info(df_nans.copy())

    assert isinstance(formatted_df, pd.DataFrame)

    assert 'nans' not in formatted_df.columns
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


def test_accessor_same_mutual_information(df_same_mi):
    xfail_dask_and_koalas(df_same_mi)

    df_same_mi.ww.init()

    mi = df_same_mi.ww.mutual_information()

    cols_used = set(np.unique(mi[['column_1', 'column_2']].values))
    assert 'nans' not in cols_used
    assert 'nat_lang' not in cols_used
    assert mi.shape[0] == 1
    assert mi_between_cols('floats', 'ints', mi) == 1.0


def test_accessor_mutual_information(df_mi):
    xfail_dask_and_koalas(df_mi)

    df_mi.ww.init(logical_types={'dates': Datetime(datetime_format='%Y-%m-%d')})
    original_df = df_mi.copy()
    mi = df_mi.ww.mutual_information()
    assert mi.shape[0] == 10

    np.testing.assert_almost_equal(mi_between_cols('ints', 'bools', mi), 0.734, 3)
    np.testing.assert_almost_equal(mi_between_cols('ints', 'strs', mi), 0.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('strs', 'bools', mi), 0, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'ints', mi), 1.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'bools', mi), 0.734, 3)

    mi_many_rows = df_mi.ww.mutual_information(nrows=100000)
    pd.testing.assert_frame_equal(mi, mi_many_rows)

    mi = df_mi.ww.mutual_information(nrows=1)
    assert mi.shape[0] == 10
    assert (mi['mutual_info'] == 1.0).all()

    mi = df_mi.ww.mutual_information(num_bins=2)
    assert mi.shape[0] == 10
    np.testing.assert_almost_equal(mi_between_cols('bools', 'ints', mi), .274, 3)
    np.testing.assert_almost_equal(mi_between_cols('strs', 'ints', mi), 0, 3)
    np.testing.assert_almost_equal(mi_between_cols('bools', 'strs', mi), 0, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'strs', mi), 0, 3)
    np.testing.assert_almost_equal(mi_between_cols('dates', 'ints', mi), .274, 3)

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
