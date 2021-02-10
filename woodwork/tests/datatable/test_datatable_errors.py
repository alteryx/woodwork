import re

import pandas as pd
import pytest

from woodwork import DataTable
from woodwork.datatable import (
    _check_column_metadata,
    _check_index,
    _check_logical_types,
    _check_semantic_tags,
    _check_table_metadata,
    _check_time_index,
    _validate_dataframe,
    _validate_params
)
from woodwork.logical_types import Integer


def test_validate_params_errors(sample_df):
    error_message = 'Dataframe must be one of: pandas.DataFrame, dask.DataFrame, koalas.DataFrame, numpy.ndarray'
    with pytest.raises(TypeError, match=error_message):
        _validate_dataframe(dataframe=pd.Series())

    error_message = 'DataTable name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=sample_df,
                         name=1,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         table_metadata=None,
                         column_metadata=None,
                         semantic_tags=None,
                         make_index=False,
                         column_descriptions=None)


def test_check_index_errors(sample_df):
    error_message = 'Specified index column `foo` not found in dataframe. To create a new index column, set make_index to True.'
    with pytest.raises(LookupError, match=error_message):
        _check_index(dataframe=sample_df, index='foo')

    if isinstance(sample_df, pd.DataFrame):
        # Does not check for index uniqueness with Dask
        error_message = 'Index column must be unique'
        with pytest.raises(LookupError, match=error_message):
            _check_index(sample_df, index='age')

    error_message = 'When setting make_index to True, the name specified for index cannot match an existing column name'
    with pytest.raises(IndexError, match=error_message):
        _check_index(sample_df, index='id', make_index=True)

    error_message = 'When setting make_index to True, the name for the new index must be specified in the index parameter'
    with pytest.raises(IndexError, match=error_message):
        _check_index(sample_df, index=None, make_index=True)


def test_check_time_index_errors(sample_df):
    error_message = 'Specified time index column `foo` not found in dataframe'
    with pytest.raises(LookupError, match=error_message):
        _check_time_index(dataframe=sample_df, time_index='foo')


def test_check_table_metadata_errors():
    error_message = 'Table metadata must be a dictionary.'
    with pytest.raises(TypeError, match=error_message):
        _check_table_metadata('test')


def test_check_column_metadata_errors(sample_df):
    error_message = 'Column metadata must be a dictionary.'
    with pytest.raises(TypeError, match=error_message):
        _check_column_metadata(sample_df, column_metadata='test')

    column_metadata = {
        'invalid_col': {'description': 'not a valid column'}
    }
    err_msg = re.escape("column_metadata contains columns that are not present in dataframe: ['invalid_col']")
    with pytest.raises(LookupError, match=err_msg):
        DataTable(sample_df, column_metadata=column_metadata)


def test_pop_error(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types={'age': Integer},
                   semantic_tags={'age': 'custom_tag'},
                   use_standard_tags=True)

    with pytest.raises(KeyError, match="Column with name missing not found in DataTable"):
        dt.pop("missing")


def test_filter_cols_errors(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')

    filter_no_matches = dt._filter_cols(include='nothing')
    assert filter_no_matches == []


def test_datatable_drop_errors(sample_df):
    dt = DataTable(sample_df)

    error = re.escape("['not_present'] not found in DataTable")
    with pytest.raises(ValueError, match=error):
        dt.drop('not_present')

    with pytest.raises(ValueError, match=error):
        dt.drop(['age', 'not_present'])

    error = re.escape("['not_present1', 4] not found in DataTable")
    with pytest.raises(ValueError, match=error):
        dt.drop(['not_present1', 4])


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


def test_check_semantic_tags_errors(sample_df):
    error_message = 'semantic_tags must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_semantic_tags(sample_df, semantic_tags='type')

    bad_semantic_tags_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("semantic_tags contains columns that are not present in dataframe: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_semantic_tags(sample_df, bad_semantic_tags_keys)
