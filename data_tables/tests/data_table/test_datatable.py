import re

import pandas as pd
import pytest

from data_tables import DataTable
from data_tables.data_table import (
    _check_index,
    _check_logical_types,
    _check_time_index,
    _check_unique_column_names,
    _validate_params
)
from data_tables.logical_types import Double, NaturalLanguage


def test_datatable_init(sample_df):
    dt = DataTable(sample_df)
    df = dt.df

    assert dt.name is None
    assert dt.index is None
    assert dt.time_index is None
    assert isinstance(df, pd.DataFrame)
    assert set(dt.columns.keys()) == set(sample_df.columns)
    assert df is sample_df
    pd.testing.assert_frame_equal(df, sample_df)


def test_datatable_copy_param(sample_df):
    dt_with_copy = DataTable(sample_df, copy_dataframe=True)
    assert sample_df is not dt_with_copy.df

    dt_no_copy = DataTable(sample_df)
    assert sample_df is dt_no_copy.df


def test_datatable_init_with_name_and_index_vals(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   index='id',
                   time_index='signup_date')

    assert dt.name == 'datatable'
    assert dt.index == 'id'
    assert dt.time_index == 'signup_date'


def test_datatable_init_with_logical_types(sample_df):
    logical_types = {
        'full_name': NaturalLanguage,
        'age': Double
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types=logical_types)
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == Double


def test_validate_params_errors(sample_df):
    error_message = 'Dataframe must be a pandas.DataFrame'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=pd.Series(),
                         name=None,
                         index=None,
                         time_index=None,
                         logical_types=None)

    error_message = 'DataTable name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=sample_df,
                         name=1,
                         index=None,
                         time_index=None,
                         logical_types=None)


def test_check_index_errors(sample_df):
    error_message = 'Index column name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _check_index(dataframe=sample_df, index=1)

    error_message = 'Specified index column `foo` not found in dataframe'
    with pytest.raises(LookupError, match=error_message):
        _check_index(dataframe=sample_df, index='foo')

    error_message = 'Index column must be unique'
    with pytest.raises(LookupError, match=error_message):
        _check_index(sample_df, index='age')


def test_check_time_index_errors(sample_df):
    error_message = 'Time index column name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _check_time_index(dataframe=sample_df,
                          time_index=1)

    error_message = 'Specified time index column `foo` not found in dataframe'
    with pytest.raises(LookupError, match=error_message):
        _check_time_index(dataframe=sample_df, time_index='foo')


def test_check_unique_column_names(sample_df):
    duplicate_cols_df = sample_df.copy()
    duplicate_cols_df.insert(0, 'age', [18, 21, 65], allow_duplicates=True)
    with pytest.raises(IndexError, match='Dataframe cannot contain duplicate columns names'):
        _check_unique_column_names(duplicate_cols_df)


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
