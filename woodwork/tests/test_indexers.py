import pandas as pd
import pytest

import woodwork as ww
from woodwork import DataColumn, DataTable
from woodwork.indexers import _iLocIndexer
from woodwork.logical_types import (
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    Integer,
    PhoneNumber
)
from woodwork.tests.testing_utils import to_pandas
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')


def test_iLocIndexer_class_error(sample_df_dask, sample_series_dask):
    dt_dask = ww.DataTable(sample_df_dask)
    with pytest.raises(TypeError, match="iloc is not supported for Dask DataTables"):
        _iLocIndexer(dt_dask)

    dc_dask = ww.DataColumn(sample_series_dask)
    with pytest.raises(TypeError, match="iloc is not supported for Dask DataColumns"):
        _iLocIndexer(dc_dask)


def test_iLocIndexer_class(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('iloc is not supported with Dask inputs')
    dt = ww.DataTable(sample_df)
    ind = _iLocIndexer(dt)
    pd.testing.assert_frame_equal(to_pandas(ind.underlying_data), to_pandas(sample_df))
    pd.testing.assert_frame_equal(to_pandas(ind[1:2].to_dataframe()), to_pandas(sample_df.iloc[1:2]))
    assert ind[0, 0] == 0


def test_iloc_datacolumn(sample_series):
    if dd and isinstance(sample_series, dd.Series):
        pytest.xfail('iloc is not supported with Dask inputs')
    semantic_tags = ['tag1', 'tag2']
    logical_type = Categorical
    dcol = DataColumn(sample_series, logical_type=logical_type, semantic_tags=semantic_tags)
    sliced = dcol.iloc[2:]

    assert sliced.name == "sample_series"
    assert sliced.logical_type == logical_type
    assert sliced.semantic_tags == {'category', 'tag1', 'tag2'}
    pd.testing.assert_series_equal(to_pandas(sliced._series), to_pandas(dcol._series.iloc[2:]))

    assert dcol.iloc[0] == 'a'

    dcol_no_std_tags = DataColumn(sample_series, use_standard_tags=False)
    sliced = dcol_no_std_tags.iloc[:]
    assert sliced.name
    assert sliced.logical_type == logical_type
    assert sliced.semantic_tags == set()


def test_iloc_indices_column(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('iloc is not supported with Dask inputs')
    dt_indices = DataTable(sample_df, index='id', time_index='signup_date')
    sliced_index = dt_indices.iloc[:, 0]
    assert sliced_index.semantic_tags == {'numeric'}

    sliced_time_index = dt_indices.iloc[:, 5]
    assert sliced_time_index.semantic_tags == set()


def test_iloc_with_properties(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('iloc is not supported with Dask inputs')
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'phone_number': ['tag3', 'tag2'],
        'signup_date': {'secondary_time_index'},
    }
    logical_types = {
        'full_name': Categorical,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
    }
    dt = DataTable(sample_df, logical_types=logical_types, semantic_tags=semantic_tags)
    sliced = dt.iloc[1:3, 1:3]
    assert sliced.shape == (2, 2)

    assert sliced.semantic_tags == {'full_name': {'category', 'tag1'}, 'email': {'tag2'}}
    assert sliced.logical_types == {'full_name': Categorical, 'email': EmailAddress}
    assert sliced.index is None

    dt_no_std_tags = DataTable(sample_df, logical_types=logical_types, use_standard_tags=False)
    sliced = dt_no_std_tags.iloc[:, [0, 5]]
    assert sliced.semantic_tags == {'id': set(), 'signup_date': set()}
    assert sliced.logical_types == {'id': Integer, 'signup_date': Datetime}
    assert sliced.index is None


def test_iloc_dimensionality(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('iloc is not supported with Dask inputs')
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'phone_number': ['tag3', 'tag2'],
        'signup_date': {'secondary_time_index'},
    }
    logical_types = {
        'full_name': Categorical,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
    }
    dt = DataTable(sample_df, logical_types=logical_types, semantic_tags=semantic_tags)

    sliced_series_row = dt.iloc[1]
    assert isinstance(sliced_series_row, pd.Series)
    assert set(sliced_series_row.index) == set(sample_df.columns)
    assert sliced_series_row.name == 1

    sliced_series_col = dt.iloc[:, 1]
    assert sliced_series_col.logical_type == Categorical
    assert sliced_series_col.semantic_tags == {'tag1', 'category'}
    assert sliced_series_col.name == 'full_name'


def test_iloc_indices(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('iloc is not supported with Dask inputs')
    dt_with_index = DataTable(sample_df, index='id')
    assert dt_with_index.iloc[:, [0, 5]].index == 'id'
    assert dt_with_index.iloc[:, [1, 2]].index is None

    dt_with_time_index = DataTable(sample_df, time_index='signup_date')
    assert dt_with_time_index.iloc[:, [0, 5]].time_index == 'signup_date'
    assert dt_with_time_index.iloc[:, [1, 2]].index is None
