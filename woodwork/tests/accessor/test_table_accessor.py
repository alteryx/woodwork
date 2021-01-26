import pytest
import re

import pandas as pd

from woodwork.logical_types import Datetime, Double, Integer, FullName, NaturalLanguage
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


def test_accessor_separation_of_params(sample_df):
    xfail_dask_and_koalas(sample_df)
    # mix up order of acccessor and schema params
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_name', index='id', semantic_tags={'id': 'test_tag'}, time_index='signup_date')

    assert schema_df.ww.semantic_tags['id'] == {'index', 'test_tag'}
    assert schema_df.ww.index == 'id'
    assert schema_df.ww.time_index == 'signup_date'
    assert schema_df.ww.name == 'test_name'


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


def test_accessor_init_with_valid_string_time_index(time_index_df):
    xfail_dask_and_koalas(time_index_df)

    time_index_df.ww.init(name='schema',
                          index='id',
                          time_index='times')

    assert time_index_df.ww.name == 'schema'
    assert time_index_df.ww.index == 'id'
    assert time_index_df.ww.time_index == 'times'
    assert time_index_df.ww.columns[time_index_df.ww.time_index]['logical_type'] == Datetime


def test_accessor_init_with_numeric_datetime_time_index(time_index_df):
    xfail_dask_and_koalas(time_index_df)
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints', logical_types={'ints': Datetime})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        time_index_df.ww.init(name='schema', time_index='strs', logical_types={'strs': Datetime})

    assert schema_df.ww.time_index == 'ints'
    assert schema_df['ints'].dtype == 'datetime64[ns]'


def test_accessor_with_numeric_time_index(time_index_df):
    xfail_dask_and_koalas(time_index_df)
    # Set a numeric time index on init
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints')
    date_col = schema_df.ww.columns['ints']
    assert schema_df.ww.time_index == 'ints'
    assert date_col['logical_type'] == Integer
    assert date_col['semantic_tags'] == {'time_index', 'numeric'}

    # Specify logical type for time index on init
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints', logical_types={'ints': 'Double'})
    date_col = schema_df.ww.columns['ints']
    assert schema_df.ww.time_index == 'ints'
    assert date_col['logical_type'] == Double
    assert date_col['semantic_tags'] == {'time_index', 'numeric'}

    # --> add back when schema updates are implemented
    # # Change time index to normal datetime time index
    # schema = schema.set_time_index('times')
    # date_col = schema['ints']
    # assert schema.time_index == 'times'
    # assert date_col.logical_type == Double
    # assert date_col.semantic_tags == {'numeric'}

    # Set numeric time index after init
    # schema = Schema(time_index_df, logical_types={'ints': 'Double'})
    # schema = schema.set_time_index('ints')
    # date_col = schema['ints']
    # assert schema.time_index == 'ints'
    # assert date_col.logical_type == Double
    # assert date_col.semantic_tags == {'time_index', 'numeric'}


def test_accessor_init_with_invalid_string_time_index(sample_df):
    xfail_dask_and_koalas(sample_df)
    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        sample_df.ww.init(name='schema', time_index='full_name')


def test_accessor_init_with_string_logical_types(sample_df):
    xfail_dask_and_koalas(sample_df)
    logical_types = {
        'full_name': 'natural_language',
        'age': 'Double'
    }
    schema_df = sample_df.copy()
    schema_df.ww.init(name='schema',
                      logical_types=logical_types)
    assert schema_df.ww.columns['full_name']['logical_type'] == NaturalLanguage
    assert schema_df.ww.columns['age']['logical_type'] == Double

    logical_types = {
        'full_name': 'NaturalLanguage',
        'age': 'Integer',
        'signup_date': 'Datetime'
    }
    schema_df = sample_df.copy()
    schema_df.ww.init(name='schema',
                      logical_types=logical_types,
                      time_index='signup_date'
                      )
    assert schema_df.ww.columns['full_name']['logical_type'] == NaturalLanguage
    assert schema_df.ww.columns['age']['logical_type'] == Integer
    assert schema_df.ww.time_index == 'signup_date'


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
    assert schema_df['id'].dtype == 'Int64'
    assert schema_df['full_name'].dtype == 'string'


def test_underlying_index_no_index(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    assert type(sample_df.index) == pd.RangeIndex

    schema_df = sample_df.copy()
    schema_df.ww.init()
    assert type(schema_df.index) == pd.RangeIndex

    sample_df = sample_df.sort_values('full_name')
    assert type(sample_df.index) == pd.Int64Index
    sample_df.ww.init()
    assert type(sample_df.index) == pd.RangeIndex


def test_underlying_index(sample_df):
    if dd and isinstance(sample_df, dd.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    specified_index = pd.Index

    schema_df = sample_df.copy()
    schema_df.ww.init(index='full_name')
    assert schema_df.index.name is None
    assert (schema_df.index == ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown', 'Ms. Paige Turner']).all()
    assert type(schema_df.index) == specified_index

    # --> add back when schema updates are implemented
    # schema = Schema(sample_df.copy())
    # schema = schema.set_index('full_name')
    # assert (schema._dataframe.index == schema.to_dataframe()['full_name']).all()
    # assert schema._dataframe.index.name is None
    # assert type(schema._dataframe.index) == specified_index
    # assert type(schema.to_dataframe().index) == specified_index

    # schema.index = 'id'
    # assert (schema._dataframe.index == [0, 1, 2, 3]).all()
    # assert schema._dataframe.index.name is None
    # assert type(schema._dataframe.index) == specified_index
    # assert type(schema.to_dataframe().index) == specified_index

    # # test removing index removes the dataframe's index
    # schema.index = None
    # assert type(schema._dataframe.index) == unspecified_index
    # assert type(schema.to_dataframe().index) == unspecified_index

    schema_df = sample_df.copy()
    schema_df.ww.init(index='made_index', make_index=True)
    assert (schema_df.index == [0, 1, 2, 3]).all()
    assert schema_df.index.name is None
    assert type(schema_df.index) == specified_index

    # --> add back when schema updates are implemented
    # schema_dropped = schema.drop('made_index')
    # assert 'made_index' not in schema_dropped.columns
    # assert 'made_index' not in schema_dropped._dataframe.columns
    # assert type(schema_dropped._dataframe.index) == unspecified_index
    # assert type(schema_dropped.to_dataframe().index) == unspecified_index


def test_accessor_already_sorted(sample_unsorted_df):
    if dd and isinstance(sample_unsorted_df, dd.DataFrame):
        pytest.xfail('Sorting dataframe is not supported with Dask input')
    if ks and isinstance(sample_unsorted_df, ks.DataFrame):
        pytest.xfail('Sorting dataframe is not supported with Koalas input')

    schema_df = sample_unsorted_df.copy()
    schema_df.ww.init(name='schema',
                      index='id',
                      time_index='signup_date')

    assert schema_df.ww.time_index == 'signup_date'
    assert schema_df.ww.columns[schema_df.ww.time_index]['logical_type'] == Datetime

    sorted_df = to_pandas(sample_unsorted_df).sort_values(['signup_date', 'id']).set_index('id', drop=False)
    sorted_df.index.name = None
    pd.testing.assert_frame_equal(sorted_df,
                                  to_pandas(schema_df), check_index_type=False, check_dtype=False)

    schema_df = sample_unsorted_df.copy()
    schema_df.ww.init(name='schema',
                      index='id',
                      time_index='signup_date',
                      already_sorted=True)

    assert schema_df.ww.time_index == 'signup_date'
    assert schema_df.ww.columns[schema_df.ww.time_index]['logical_type'] == Datetime

    unsorted_df = to_pandas(sample_unsorted_df.set_index('id', drop=False))
    unsorted_df.index.name = None
    pd.testing.assert_frame_equal(unsorted_df, to_pandas(schema_df), check_index_type=False, check_dtype=False)
