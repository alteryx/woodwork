import re

import pandas as pd
import pytest

from data_tables import DataTable
from data_tables.data_table import (
    _check_index,
    _check_logical_types,
    _check_semantic_types,
    _check_time_index,
    _check_unique_column_names,
    _validate_params
)
from data_tables.logical_types import (
    Datetime,
    Double,
    EmailAddress,
    FullName,
    Integer,
    NaturalLanguage,
    PhoneNumber
)


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


def test_datatable_init_with_semantic_types(sample_df):
    semantic_types = {
        'id': {'index': {}},
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   semantic_types=semantic_types)

    id_semantic_types = dt.columns['id'].semantic_types
    assert isinstance(id_semantic_types, dict)
    assert len(id_semantic_types) == 1
    assert 'index' in id_semantic_types.keys()
    assert id_semantic_types['index'] == dict()


def test_validate_params_errors(sample_df):
    error_message = 'Dataframe must be a pandas.DataFrame'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=pd.Series(),
                         name=None,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         semantic_types=None)

    error_message = 'DataTable name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=sample_df,
                         name=1,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         semantic_types=None)


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


def test_check_semantic_types_errors(sample_df):
    error_message = 'semantic_types must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_semantic_types(sample_df, semantic_types='type')

    bad_semantic_types_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("semantic_types contains columns that are not present in dataframe: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_semantic_types(sample_df, bad_semantic_types_keys)


def test_set_logical_types(sample_df):
    semantic_types = {
        'full_name': {'tag1': {}},
        'email': {'tag2': {'option1': 'value1'}},
        'phone_number': {'tag3': {}},
        'signup_date': {'secondary_time_index': {'columns': ['expired']}},
    }
    dt = DataTable(sample_df, semantic_types=semantic_types)
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['email'].logical_type == NaturalLanguage
    assert dt.columns['phone_number'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == Integer
    assert dt.columns['signup_date'].logical_type == Datetime
    original_name_column = dt.columns['full_name']

    dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Double,
    })

    assert dt.columns['full_name'].logical_type == FullName
    assert dt.columns['email'].logical_type == EmailAddress
    assert dt.columns['phone_number'].logical_type == PhoneNumber
    assert dt.columns['age'].logical_type == Double
    assert dt.columns['signup_date'].logical_type == Double

    # Verify new column object was created
    new_name_column = dt.columns['full_name']
    assert new_name_column != original_name_column

    # Verify semantic types were not changed
    assert dt.columns['full_name'].semantic_types == semantic_types['full_name']
    assert dt.columns['email'].semantic_types == semantic_types['email']
    assert dt.columns['phone_number'].semantic_types == semantic_types['phone_number']
    assert dt.columns['age'].semantic_types == dict()
    assert dt.columns['signup_date'].semantic_types == semantic_types['signup_date']


def test_set_logical_types_invalid_data(sample_df):
    dt = DataTable(sample_df)
    error_message = re.escape("logical_types contains columns that are not present in dataframe: ['birthday']")
    with pytest.raises(LookupError, match=error_message):
        dt.set_logical_types({'birthday': Double})

    error_message = "Invalid logical type specified for 'age'"
    with pytest.raises(TypeError, match=error_message):
        dt.set_logical_types({'age': int})
