import pytest
import re

import pandas as pd

from woodwork.logical_types import Integer, FullName
from woodwork.schema import Schema
from woodwork.table_accessor import (
    _check_index,
    _check_logical_types,
    _check_time_index,
    _check_unique_column_names,
    _validate_accessor_params,
    _validate_schema_params
)
from woodwork.tests.testing_utils import to_pandas
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def xfail_dask_and_koalas(df):
    if dd and isinstance(df, dd.DataFrame) or ks and isinstance(df, ks.DataFrame):
        pytest.xfail('Dask and Koalas Accessors not yet supported.')


def test_validate_schema_params_errors():
    error = 'Parameter not_present does not exist on the Schema class.'
    with pytest.raises(TypeError, match=error):
        _validate_schema_params({
            'name': 'schema',
            'semantic_tags': {'id': 'tag'},
            'not_present': True
        })


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


def test_check_time_index_errors(sample_df):
    error_message = 'Specified time index column `foo` not found in dataframe'
    with pytest.raises(LookupError, match=error_message):
        _check_time_index(dataframe=sample_df, time_index='foo')


def test_check_unique_column_names_errors(sample_df):
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.skip("Koalas enforces unique column names")
    duplicate_cols_df = sample_df.copy()
    if dd and isinstance(sample_df, dd.DataFrame):
        duplicate_cols_df = dd.concat([duplicate_cols_df, duplicate_cols_df['age']], axis=1)
    else:
        duplicate_cols_df.insert(0, 'age', [18, 21, 65, 43], allow_duplicates=True)
    with pytest.raises(IndexError, match='Dataframe cannot contain duplicate columns names'):
        _check_unique_column_names(duplicate_cols_df)


def test_accessor_init(sample_df):
    xfail_dask_and_koalas(sample_df)

    assert sample_df.ww.schema is None
    sample_df.ww.init()
    assert isinstance(sample_df.ww.schema, Schema)


def test_accessor_getattr(sample_df):
    xfail_dask_and_koalas(sample_df)

    sample_df.ww.init()

    assert sample_df.ww.name is None
    assert sample_df.ww.index is None
    assert sample_df.ww.time_index is None

    assert set(sample_df.ww.columns.keys()) == set(sample_df.columns)

    # --> do test to make sure dataframe obj is the same????


def test_accessor_attr_precedence(sample_df):
    # --> will have to wait until we have an attr that matches the schema and accessor
    pass


def test_make_index(sample_df):
    xfail_dask_and_koalas(sample_df)

    schema_df = sample_df.copy()
    schema_df.ww.init(index='new_index', make_index=True)

    assert schema_df.ww.index == 'new_index'
    assert 'new_index' in schema_df.ww.columns
    assert 'new_index' in schema_df.ww.columns
    assert to_pandas(schema_df)['new_index'].unique
    assert to_pandas(schema_df['new_index']).is_monotonic
    assert 'index' in schema_df.ww.columns['new_index']['semantic_tags']


def test_accessor_logical_types(sample_df):
    # --> need much more thoughrough testing of all of this - from datatable!
    xfail_dask_and_koalas(sample_df)

    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types={'full_name': 'FullName'})

    assert schema_df.ww.logical_types['id'] == Integer
    assert schema_df.ww.logical_types['full_name'] == FullName
