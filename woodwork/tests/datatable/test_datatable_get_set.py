import re

import numpy as np
import pandas as pd
import pytest

from woodwork import DataColumn, DataTable
from woodwork.exceptions import ColumnNameMismatchWarning
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Double,
    Integer,
    ZIPCode
)
from woodwork.tests.datatable.test_datatable import ks
from woodwork.tests.testing_utils import to_pandas


def test_getitem(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types={'age': Integer},
                   semantic_tags={'age': 'custom_tag'},
                   use_standard_tags=True)

    data_col = dt['age']
    assert isinstance(data_col, DataColumn)
    assert data_col.logical_type == Integer
    assert data_col.semantic_tags == {'numeric', 'custom_tag'}


def test_getitem_invalid_input(sample_df):
    dt = DataTable(sample_df)

    error_msg = 'Column with name 1 not found in DataTable'
    with pytest.raises(KeyError, match=error_msg):
        dt[1]

    error_msg = "Column with name invalid_column not found in DataTable"
    with pytest.raises(KeyError, match=error_msg):
        dt['invalid_column']


def test_datatable_getitem_list_input(sample_df):
    # Test regular columns
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    df = dt.to_dataframe()
    columns = ['age', 'full_name']
    new_dt = dt[columns]
    assert new_dt is not dt
    assert new_dt.to_dataframe() is not df
    pd.testing.assert_frame_equal(to_pandas(df[columns]), to_pandas(new_dt.to_dataframe()))
    assert all(new_dt.to_dataframe().columns == ['age', 'full_name'])
    assert set(new_dt.columns.keys()) == {'age', 'full_name'}
    assert new_dt.index is None
    assert new_dt.time_index is None

    # Test with index
    columns = ['id', 'full_name']
    new_dt = dt[columns]
    assert new_dt is not dt
    assert new_dt.to_dataframe() is not df
    pd.testing.assert_frame_equal(to_pandas(df[columns]), to_pandas(new_dt.to_dataframe()))
    assert all(new_dt.to_dataframe().columns == ['id', 'full_name'])
    assert set(new_dt.columns.keys()) == {'id', 'full_name'}
    assert new_dt.index == 'id'
    assert new_dt.time_index is None

    # Test with time_index
    columns = ['id', 'signup_date', 'full_name']
    new_dt = dt[columns]
    assert new_dt is not dt
    assert new_dt.to_dataframe() is not df
    pd.testing.assert_frame_equal(to_pandas(df[columns]), to_pandas(new_dt.to_dataframe()), check_index_type=False)
    assert all(new_dt.to_dataframe().columns == ['id', 'signup_date', 'full_name'])
    assert set(new_dt.columns.keys()) == {'id', 'signup_date', 'full_name'}
    assert new_dt.index == 'id'

    # Test with empty list selector
    columns = []
    new_dt = dt[columns]
    assert new_dt is not dt
    assert new_dt.to_dataframe() is not df
    assert to_pandas(new_dt.to_dataframe()).empty
    assert set(new_dt.columns.keys()) == set()
    assert new_dt.index is None
    assert new_dt.time_index is None

    # Test that reversed column order reverses resulting column order
    columns = list(reversed(list(dt.columns.keys())))
    new_dt = dt[columns]

    assert new_dt is not dt
    assert new_dt.to_dataframe() is not df
    assert all(df.columns[::-1] == new_dt.to_dataframe().columns)
    assert all(dt.types.index[::-1] == new_dt.types.index)
    assert all(new_dt.to_dataframe().columns == new_dt.types.index)
    assert set(new_dt.columns.keys()) == set(dt.columns.keys())
    assert new_dt.index == 'id'
    assert new_dt.time_index == 'signup_date'


def test_datatable_getitem_list_warnings(sample_df):
    # Test regular columns
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    columns = ['age', 'invalid_col1', 'invalid_col2']
    error_msg = re.escape("Column(s) 'invalid_col1, invalid_col2' not found in DataTable")
    with pytest.raises(KeyError, match=error_msg):
        dt[columns]


def test_setitem_invalid_input(sample_df):
    dt = DataTable(sample_df, index='id', time_index='signup_date')

    error_msg = 'New column must be of DataColumn type'
    with pytest.raises(ValueError, match=error_msg):
        dt['test'] = pd.Series([1, 2, 3], dtype='Int64')

    error_msg = 'Cannot reassign index. Change column name and then use dt.set_index to reassign index.'
    with pytest.raises(KeyError, match=error_msg):
        dt['id'] = DataColumn(pd.Series([True, False, False]))

    error_msg = 'Cannot reassign time index. Change column name and then use dt.set_time_index to reassign time index.'
    with pytest.raises(KeyError, match=error_msg):
        dt['signup_date'] = DataColumn(pd.Series(['test text', 'file', 'False']))


def test_setitem_different_name(sample_df):
    dt = DataTable(sample_df)

    new_series = pd.Series([1, 2, 3, 4], name='wrong')
    if ks and isinstance(sample_df, ks.DataFrame):
        new_series = ks.Series(new_series)

    warning = 'Name mismatch between wrong and id. DataColumn and underlying series name are now id'
    with pytest.warns(ColumnNameMismatchWarning, match=warning):
        dt['id'] = DataColumn(new_series,
                              use_standard_tags=False)

    assert dt['id'].name == 'id'
    assert dt['id'].to_series().name == 'id'
    assert dt.to_dataframe()['id'].name == 'id'
    assert 'wrong' not in dt.columns

    new_series2 = pd.Series([1, 2, 3, 4], name='wrong2')
    if ks and isinstance(sample_df, ks.DataFrame):
        new_series2 = ks.Series(new_series2)

    warning = 'Name mismatch between wrong2 and new_col. DataColumn and underlying series name are now new_col'
    with pytest.warns(ColumnNameMismatchWarning, match=warning):
        dt['new_col'] = DataColumn(new_series2,
                                   use_standard_tags=False)

    assert dt['new_col'].name == 'new_col'
    assert dt['new_col'].to_series().name == 'new_col'
    assert dt.to_dataframe()['new_col'].name == 'new_col'
    assert 'wrong2' not in dt.columns

    warning = 'Name mismatch between wrong and col_with_name. DataColumn and underlying series name are now col_with_name'
    with pytest.warns(ColumnNameMismatchWarning, match=warning):
        dt['col_with_name'] = DataColumn(new_series,
                                         use_standard_tags=False, name='wrong')
    assert dt['col_with_name'].name == 'col_with_name'
    assert dt['col_with_name'].to_series().name == 'col_with_name'
    assert dt.to_dataframe()['col_with_name'].name == 'col_with_name'
    assert 'wrong' not in dt.columns


def test_setitem_new_column(sample_df):
    dt = DataTable(sample_df)
    new_series = pd.Series([1, 2, 3])
    if ks and isinstance(sample_df, ks.DataFrame):
        dtype = 'int64'
        new_series = ks.Series(new_series)
    else:
        dtype = 'Int64'

    new_col = DataColumn(new_series, use_standard_tags=False)
    assert new_col.name is None

    dt['test_col2'] = new_col
    updated_df = dt.to_dataframe()
    assert 'test_col2' in dt.columns
    assert dt['test_col2'].logical_type == Integer
    assert dt['test_col2'].semantic_tags == set()
    assert dt['test_col2'].name == 'test_col2'
    assert dt['test_col2']._series.name == 'test_col2'
    assert 'test_col2' in updated_df.columns
    assert updated_df['test_col2'].dtype == dtype

    # Standard tags and no logical type
    new_series = pd.Series(['new', 'column', 'inserted'], name='test_col')
    if ks and isinstance(sample_df, ks.DataFrame):
        dtype = 'object'
        new_series = ks.Series(new_series)
    else:
        dtype = 'category'
    new_col = DataColumn(new_series, use_standard_tags=True)
    dt['test_col'] = new_col
    updated_df = dt.to_dataframe()
    assert 'test_col' in dt.columns
    assert dt['test_col'].logical_type == Categorical
    assert dt['test_col'].semantic_tags == {'category'}
    assert dt['test_col'].name == 'test_col'
    assert dt['test_col']._series.name == 'test_col'
    assert 'test_col' in updated_df.columns
    assert updated_df['test_col'].dtype == dtype

    # Add with logical type and semantic tag
    new_series = pd.Series([1, 2, 3])
    if ks and isinstance(sample_df, ks.DataFrame):
        new_series = ks.Series(new_series)
    new_col = DataColumn(new_series,
                         logical_type=Double,
                         use_standard_tags=False,
                         semantic_tags={'test_tag'})
    dt['test_col3'] = new_col
    updated_df = dt.to_dataframe()
    assert 'test_col3' in dt.columns
    assert dt['test_col3'].logical_type == Double
    assert dt['test_col3'].semantic_tags == {'test_tag'}
    assert dt['test_col3'].name == 'test_col3'
    assert dt['test_col3']._series.name == 'test_col3'
    assert 'test_col3' in updated_df.columns
    assert updated_df['test_col3'].dtype == 'float'


def test_setitem_overwrite_column(sample_df):
    dt = DataTable(sample_df, index='id',
                   time_index='signup_date',
                   use_standard_tags=True)

    # Change to column no change in types
    original_col = dt['age']
    new_series = pd.Series([1, 2, 3])
    if ks and isinstance(sample_df, ks.DataFrame):
        dtype = 'int64'
        new_series = ks.Series(new_series)
    else:
        dtype = 'Int64'
    overwrite_col = DataColumn(new_series, use_standard_tags=True)
    dt['age'] = overwrite_col
    updated_df = dt.to_dataframe()

    assert 'age' in dt.columns
    assert dt['age'].logical_type == original_col.logical_type
    assert dt['age'].semantic_tags == original_col.semantic_tags
    assert 'age' in updated_df.columns
    assert updated_df['age'].dtype == dtype
    assert original_col.to_series() is not dt['age'].to_series()

    # Change dtype, logical types, and tags with conflicting use_standard_tags
    original_col = dt['full_name']
    new_series = pd.Series([True, False, False])
    if ks and isinstance(sample_df, ks.DataFrame):
        new_series = ks.Series(new_series)
        dtype = 'bool'
    else:
        dtype = 'boolean'
    overwrite_col = DataColumn(new_series.astype(dtype),
                               use_standard_tags=False,
                               semantic_tags='test_tag')
    dt['full_name'] = overwrite_col
    updated_df = dt.to_dataframe()

    assert 'full_name' in dt.columns
    assert dt['full_name'].logical_type == Boolean
    assert dt['full_name'].semantic_tags == {'test_tag'}
    assert 'full_name' in updated_df.columns
    assert updated_df['full_name'].dtype == dtype
    assert original_col.to_series() is not dt['full_name'].to_series()


def test_setitem_with_differnt_types(sample_df_pandas):
    dt = DataTable(sample_df_pandas)

    dt['np_array_col'] = DataColumn(np.array([1, 3, 4, 5]))
    assert 'np_array_col' in dt.columns
    assert 'np_array_col' in dt._dataframe.columns
    assert dt['np_array_col'].name == 'np_array_col'
    assert isinstance(dt['np_array_col']._series, pd.Series)

    dt['extension_col'] = DataColumn(pd.Categorical(['a', 'b', 'c', 'd']), logical_type='ZipCode', name='extension_col')
    assert 'extension_col' in dt.columns
    assert 'extension_col' in dt._dataframe.columns
    assert dt['extension_col'].name == 'extension_col'
    assert isinstance(dt['extension_col']._series, pd.Series)
    assert dt['extension_col'].logical_type == ZIPCode
