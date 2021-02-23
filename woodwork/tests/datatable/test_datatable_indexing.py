import pandas as pd
import pytest

from woodwork import DataTable
from woodwork.logical_types import Double, Integer
from woodwork.tests.datatable.test_datatable import dd, ks
from woodwork.tests.testing_utils import to_pandas


def test_pop_index(sample_df):
    dt = DataTable(sample_df, index='id', name='dt_name')
    assert dt.index == 'id'
    id_col = dt.pop('id')
    assert dt.index is None
    assert 'index' in id_col.semantic_tags


def test_set_index(sample_df):
    # Test setting index with set_index()
    dt = DataTable(sample_df)
    new_dt = dt.set_index('id')
    assert new_dt is not dt
    assert new_dt.index == 'id'
    assert dt.index is None
    assert new_dt.columns['id'].semantic_tags == {'index'}
    non_index_cols = [col for col in new_dt.columns.values() if col.name != 'id']
    assert all(['index' not in col.semantic_tags for col in non_index_cols])
    # Test changing index with set_index()
    new_dt2 = new_dt.set_index('full_name')
    assert new_dt.index == 'id'
    assert new_dt2.columns['full_name'].semantic_tags == {'index'}
    non_index_cols = [col for col in new_dt2.columns.values() if col.name != 'full_name']
    assert all(['index' not in col.semantic_tags for col in non_index_cols])

    # Test setting index using setter
    dt = DataTable(sample_df)
    dt.index = 'id'
    assert dt.index == 'id'
    assert 'index' in dt.columns['id'].semantic_tags
    non_index_cols = [col for col in dt.columns.values() if col.name != 'id']
    assert all(['index' not in col.semantic_tags for col in non_index_cols])
    # Test changing index with setter
    dt.index = 'full_name'
    assert 'index' in dt.columns['full_name'].semantic_tags
    non_index_cols = [col for col in dt.columns.values() if col.name != 'full_name']
    assert all(['index' not in col.semantic_tags for col in non_index_cols])

    # Test changing index also changes underlying DataFrame - pandas only
    if isinstance(sample_df, pd.DataFrame):
        dt = DataTable(sample_df)
        dt.index = 'id'
        assert (dt.to_dataframe().index == [0, 1, 2, 3]).all()
        assert (dt._dataframe.index == [0, 1, 2, 3]).all()
        dt.index = 'full_name'
        assert (dt.to_dataframe().index == dt.to_dataframe()['full_name']).all()
        assert (dt._dataframe.index == dt.to_dataframe()['full_name']).all()


def test_set_index_twice(sample_df):
    dt = DataTable(sample_df, index='id', time_index='signup_date')
    original_df = dt.df.copy()

    dt_index_twice = dt.set_index('id')
    assert 'index' in dt_index_twice['id'].semantic_tags
    assert dt_index_twice.index == 'id'
    assert dt_index_twice == dt
    pd.testing.assert_frame_equal(to_pandas(original_df), to_pandas(dt_index_twice.df))

    dt_time_index_twice = dt.set_time_index('signup_date')
    assert 'time_index' in dt_time_index_twice['signup_date'].semantic_tags
    assert dt_time_index_twice.time_index == 'signup_date'
    assert dt_time_index_twice == dt
    pd.testing.assert_frame_equal(to_pandas(original_df), to_pandas(dt_time_index_twice.df))

    dt.index = 'id'
    assert 'index' in dt['id'].semantic_tags
    assert dt.index == 'id'
    pd.testing.assert_frame_equal(to_pandas(original_df), to_pandas(dt.df))

    dt.time_index = 'signup_date'
    assert 'time_index' in dt['signup_date'].semantic_tags
    assert dt.time_index == 'signup_date'
    pd.testing.assert_frame_equal(to_pandas(original_df), to_pandas(dt.df))


def test_underlying_index_no_index(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    assert type(sample_df.index) == pd.RangeIndex
    dt = DataTable(sample_df.copy())
    assert type(dt._dataframe.index) == pd.RangeIndex
    assert type(dt.to_dataframe().index) == pd.RangeIndex

    sample_df = sample_df.sort_values('full_name')
    assert type(sample_df.index) == pd.Int64Index
    dt = DataTable(sample_df)

    assert type(dt._dataframe.index) == pd.Int64Index
    assert type(dt.to_dataframe().index) == pd.Int64Index


def test_underlying_index(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    unspecified_index = pd.RangeIndex
    specified_index = pd.Index

    dt = DataTable(sample_df.copy(), index='id')
    assert dt._dataframe.index.name is None
    assert (dt._dataframe.index == [0, 1, 2, 3]).all()
    assert type(dt._dataframe.index) == specified_index
    assert type(dt.to_dataframe().index) == specified_index

    dt = DataTable(sample_df.copy())
    dt = dt.set_index('full_name')
    assert (dt._dataframe.index == dt.to_dataframe()['full_name']).all()
    assert dt._dataframe.index.name is None
    assert type(dt._dataframe.index) == specified_index
    assert type(dt.to_dataframe().index) == specified_index

    dt.index = 'id'
    assert (dt._dataframe.index == [0, 1, 2, 3]).all()
    assert dt._dataframe.index.name is None
    assert type(dt._dataframe.index) == specified_index
    assert type(dt.to_dataframe().index) == specified_index

    # test removing index removes the dataframe's index
    dt.index = None
    assert type(dt._dataframe.index) == unspecified_index
    assert type(dt.to_dataframe().index) == unspecified_index

    dt = DataTable(sample_df.copy(), index='made_index', make_index=True)
    assert (dt._dataframe.index == [0, 1, 2, 3]).all()
    assert dt._dataframe.index.name is None
    assert type(dt._dataframe.index) == specified_index
    assert type(dt.to_dataframe().index) == specified_index

    dt_dropped = dt.drop('made_index')
    assert 'made_index' not in dt_dropped.columns
    assert 'made_index' not in dt_dropped._dataframe.columns
    assert type(dt_dropped._dataframe.index) == unspecified_index
    assert type(dt_dropped.to_dataframe().index) == unspecified_index


def test_underlying_index_on_update(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    dt = DataTable(sample_df.copy(), index='id')

    dt.update_dataframe(sample_df.tail(2))
    assert (dt._dataframe.index == [2, 3]).all()
    assert dt._dataframe.index.name is None
    assert type(dt._dataframe.index) == pd.Int64Index
    assert type(dt.to_dataframe().index) == pd.Int64Index

    actual = dt.iloc[[0, 1]]
    assert type(actual._dataframe.index) == pd.Index
    assert type(actual.to_dataframe().index) == pd.Index

    actual = dt.select(dt.index)
    assert type(actual._dataframe.index) == pd.Int64Index
    assert type(actual.to_dataframe().index) == pd.Int64Index

    actual = dt[['age']]
    assert type(actual._dataframe.index) == pd.Int64Index
    assert type(actual.to_dataframe().index) == pd.Int64Index

    actual = dt.drop(dt.index)
    assert type(actual._dataframe.index) == pd.RangeIndex
    assert type(actual.to_dataframe().index) == pd.RangeIndex

    actual = dt.reset_semantic_tags(retain_index_tags=False)
    assert type(actual._dataframe.index) == pd.RangeIndex
    assert type(actual.to_dataframe().index) == pd.RangeIndex

    actual = dt.set_types(retain_index_tags=False, semantic_tags={'id': 'numeric'})
    assert type(actual._dataframe.index) == pd.RangeIndex
    assert type(actual.to_dataframe().index) == pd.RangeIndex

    dt.pop(dt.index)
    assert type(dt._dataframe.index) == pd.RangeIndex
    assert type(dt.to_dataframe().index) == pd.RangeIndex


def test_set_time_index(sample_df):
    # Test setting time index with set_time_index()
    dt = DataTable(sample_df)
    new_dt = dt.set_time_index('signup_date')
    assert new_dt is not dt
    assert dt.time_index is None
    assert new_dt.time_index == 'signup_date'
    assert 'time_index' in new_dt.columns['signup_date'].semantic_tags
    non_index_cols = [col for col in new_dt.columns.values() if col.name != 'signup_date']
    assert all(['time_index' not in col.semantic_tags for col in non_index_cols])

    # Test changing time index with set_time_index()
    sample_df['transaction_date'] = pd.to_datetime('2015-09-02')
    dt = DataTable(sample_df)
    new_dt = dt.set_time_index('signup_date')
    assert new_dt.time_index == 'signup_date'
    new_dt2 = new_dt.set_time_index('transaction_date')
    assert 'time_index' in new_dt2.columns['transaction_date'].semantic_tags
    non_index_cols = [col for col in new_dt2.columns.values() if col.name != 'transaction_date']
    assert all(['time_index' not in col.semantic_tags for col in non_index_cols])

    # Test setting index using setter
    dt = DataTable(sample_df)
    assert dt.time_index is None
    dt.time_index = 'signup_date'
    assert dt.time_index == 'signup_date'
    assert 'time_index' in dt.columns['signup_date'].semantic_tags
    non_index_cols = [col for col in dt.columns.values() if col.name != 'signup_date']
    assert all(['time_index' not in col.semantic_tags for col in non_index_cols])

    # Test changing time index with setter
    sample_df['transaction_date'] = pd.to_datetime('2015-09-02')
    dt = DataTable(sample_df)
    dt.time_index = 'signup_date'
    assert dt.time_index == 'signup_date'
    dt.time_index = 'transaction_date'
    assert 'time_index' in dt.columns['transaction_date'].semantic_tags
    non_index_cols = [col for col in dt.columns.values() if col.name != 'transaction_date']
    assert all(['time_index' not in col.semantic_tags for col in non_index_cols])


def test_datatable_clear_index(sample_df):
    # Test by removing index tag
    dt = DataTable(sample_df, index='id')
    assert dt.index == 'id'
    dt = dt.remove_semantic_tags({'id': 'index'})
    assert dt.index is None
    assert all(['index' not in col.semantic_tags for col in dt.columns.values()])

    # Test using setter
    dt = DataTable(sample_df, index='id')
    assert dt.index == 'id'
    dt.index = None
    assert dt.index is None
    assert all(['index' not in col.semantic_tags for col in dt.columns.values()])


def test_datatable_clear_time_index(sample_df):
    # Test by removing time_index tag
    dt = DataTable(sample_df, time_index='signup_date')
    assert dt.time_index == 'signup_date'
    dt = dt.remove_semantic_tags({'signup_date': 'time_index'})
    assert dt.time_index is None
    assert all(['time_index' not in col.semantic_tags for col in dt.columns.values()])

    # Test using setter
    dt = DataTable(sample_df, time_index='signup_date')
    assert dt.time_index == 'signup_date'
    dt.time_index = None
    assert dt.time_index is None
    assert all(['time_index' not in col.semantic_tags for col in dt.columns.values()])


def test_make_index(sample_df):
    dt = DataTable(sample_df, index='new_index', make_index=True)
    assert dt.index == 'new_index'
    assert 'new_index' in dt._dataframe.columns
    assert to_pandas(dt._dataframe)['new_index'].unique
    assert to_pandas(dt._dataframe['new_index']).is_monotonic
    assert 'index' in dt.columns['new_index'].semantic_tags


def test_numeric_time_index_dtypes(numeric_time_index_df):
    dt = DataTable(numeric_time_index_df, time_index='ints')
    date_col = dt['ints']
    assert dt.time_index == 'ints'
    assert date_col.logical_type == Integer
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    dt = dt.set_time_index('floats')
    date_col = dt['floats']
    assert dt.time_index == 'floats'
    assert date_col.logical_type == Double
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    dt = dt.set_time_index('with_null')
    date_col = dt['with_null']
    assert dt.time_index == 'with_null'
    if ks and isinstance(numeric_time_index_df, ks.DataFrame):
        ltype = Double
    else:
        ltype = Integer
    assert date_col.logical_type == ltype
    assert date_col.semantic_tags == {'time_index', 'numeric'}


def test_numeric_index_strings(time_index_df):
    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        DataTable(time_index_df, time_index='strs')

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        DataTable(time_index_df, time_index='ints', logical_types={'ints': 'Categorical'})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        DataTable(time_index_df, time_index='letters', logical_types={'strs': 'Integer'})

    dt = DataTable(time_index_df, time_index='strs', logical_types={'strs': 'Double'})
    date_col = dt['strs']
    assert dt.time_index == 'strs'
    assert date_col.logical_type == Double
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    dt = DataTable(time_index_df, logical_types={'strs': 'Double'})
    dt = dt.set_time_index('strs')
    date_col = dt['strs']
    assert dt.time_index == 'strs'
    assert date_col.logical_type == Double
    assert date_col.semantic_tags == {'time_index', 'numeric'}


def test_datatable_update_dataframe_with_make_index(sample_df):
    new_df = sample_df.copy().tail(2).reset_index(drop=True)
    if dd and isinstance(sample_df, dd.DataFrame):
        new_df = dd.from_pandas(new_df, npartitions=1)

    dt = DataTable(sample_df,
                   index='new_index',
                   make_index=True,
                   logical_types={'full_name': 'FullName'},
                   semantic_tags={'phone_number': 'custom_tag'})
    original_types = dt.types

    dt.update_dataframe(new_df)
    assert len(dt._dataframe) == 2
    assert dt.index == 'new_index'
    pd.testing.assert_frame_equal(original_types, dt.types)

    # confirm that DataColumn series matches corresponding dataframe column
    for col in dt.columns:
        assert to_pandas(dt.columns[col]._series).equals(to_pandas(dt._dataframe[col]))
        assert dt.columns[col]._series.dtype == dt._dataframe[col].dtype

    # confirm that we can update using current dataframe without error
    dt.update_dataframe(dt._dataframe.head(1))
    assert len(dt._dataframe) == 1


def test_datatable_drop_indices(sample_df):
    dt = DataTable(sample_df, index='id', time_index='signup_date')
    assert dt.index == 'id'
    assert dt.time_index == 'signup_date'

    dropped_index_dt = dt.drop('id')
    assert 'id' not in dropped_index_dt.columns
    assert dropped_index_dt.index is None
    assert dropped_index_dt.time_index == 'signup_date'

    dropped_time_index_dt = dt.drop(['signup_date'])
    assert 'signup_date' not in dropped_time_index_dt.columns
    assert dropped_time_index_dt.time_index is None
    assert dropped_time_index_dt.index == 'id'
