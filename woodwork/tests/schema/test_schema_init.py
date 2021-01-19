import re

import numpy as np
import pandas as pd
import pytest

import woodwork as ww
from woodwork import Schema
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
from woodwork.schema import (
    _check_column_metadata,
    _check_index,
    _check_logical_types,
    _check_semantic_tags,
    _check_table_metadata,
    _check_time_index,
    _check_unique_column_names,
    _validate_dataframe,
    _validate_params
)
from woodwork.tests.testing_utils import (
    check_column_order,
    mi_between_cols,
    to_pandas,
    validate_subset_dt
)
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
dask_delayed = import_or_none('dask.delayed')
ks = import_or_none('databricks.koalas')


def test_schema_init(sample_df):
    schema = Schema(sample_df)

    assert schema.name is None
    assert schema.index is None
    assert schema.time_index is None

    assert set(schema.columns.keys) == set(sample_df.columns)


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


def test_check_unique_column_names(sample_df):
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.skip("Koalas enforces unique column names")
    duplicate_cols_df = sample_df.copy()
    if dd and isinstance(sample_df, dd.DataFrame):
        duplicate_cols_df = dd.concat([duplicate_cols_df, duplicate_cols_df['age']], axis=1)
    else:
        duplicate_cols_df.insert(0, 'age', [18, 21, 65, 43], allow_duplicates=True)
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


def test_check_table_metadata_errors():
    error_message = 'Table metadata must be a dictionary.'
    with pytest.raises(TypeError, match=error_message):
        _check_table_metadata('test')


def test_check_column_metadata_errors(sample_df):
    error_message = 'Column metadata must be a dictionary.'
    with pytest.raises(TypeError, match=error_message):
        _check_column_metadata(sample_df, column_metadata='test')

    # --> add back in after implementing metadata on init
    # column_metadata = {
    #     'invalid_col': {'description': 'not a valid column'}
    # }
    # err_msg = re.escape("column_metadata contains columns that are not present in dataframe: ['invalid_col']")
    # with pytest.raises(LookupError, match=err_msg):
    #     Schema(sample_df, column_metadata=column_metadata)
