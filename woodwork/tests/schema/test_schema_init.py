import re

import numpy as np
import pandas as pd
import pytest

from woodwork import Schema
from woodwork.logical_types import (
    URL,
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    Filepath,
    FullName,
    Integer,
    IPAddress,
    LatLong,
    NaturalLanguage,
    Ordinal,
    PhoneNumber,
    SubRegionCode,
    ZIPCode
)
from woodwork.schema import (
    _check_column_descriptions,
    _check_column_metadata,
    _check_index,
    _check_logical_types,
    _check_semantic_tags,
    _check_table_metadata,
    _check_time_index,
    _check_column_names,
    _validate_params
)
from woodwork.tests.testing_utils import to_pandas
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
dask_delayed = import_or_none('dask.delayed')
ks = import_or_none('databricks.koalas')


def test_validate_params_errors(sample_column_names):
    error_message = 'Schema name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(column_names=sample_column_names,
                         name=1,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         table_metadata=None,
                         column_metadata=None,
                         semantic_tags=None,
                         column_descriptions=None)


def test_check_index_errors(sample_column_names):
    error_message = 'Specified index column `foo` not found in Schema.'
    with pytest.raises(LookupError, match=error_message):
        _check_index(column_names=sample_column_names, index='foo')


def test_check_time_index_errors(sample_column_names):
    error_message = 'Specified time index column `foo` not found in Schema'
    with pytest.raises(LookupError, match=error_message):
        _check_time_index(column_names=sample_column_names, time_index='foo', logical_type=Integer)

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        _check_time_index(column_names=sample_column_names, time_index='full_name', logical_type='NaturalLanguage')

    error_msg = 'String test is not a valid logical type'
    with pytest.raises(ValueError, match=error_msg):
        _check_time_index(column_names=sample_column_names, time_index='full_name', logical_type='test')


def test_check_column_names(sample_column_names):
    error_message = 'Column names must be a list or set'
    with pytest.raises(TypeError, match=error_message):
        _check_column_names(column_names=int)

    sample_column_names.append('id')
    with pytest.raises(IndexError, match='Schema cannot contain duplicate columns names'):
        _check_column_names(sample_column_names)


def test_check_logical_types_errors(sample_column_names):
    error_message = 'logical_types must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_logical_types(sample_column_names, logical_types='type')

    bad_logical_types_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("logical_types contains columns that are not present in Schema: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_logical_types(sample_column_names, bad_logical_types_keys)

    bad_logical_types_keys = {
        'id': None,
        'full_name': None,
        'email': None,
        'phone_number': None,
        'age': None,
    }
    error_message = re.escape("logical_types is missing columns that are present in Schema: ['is_registered', 'signup_date']")
    with pytest.raises(LookupError, match=error_message):
        _check_logical_types(sample_column_names, bad_logical_types_keys)


def test_check_semantic_tags_errors(sample_column_names):
    error_message = 'semantic_tags must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_semantic_tags(sample_column_names, semantic_tags='type')

    bad_semantic_tags_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("semantic_tags contains columns that are not present in Schema: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_semantic_tags(sample_column_names, bad_semantic_tags_keys)


def test_check_table_metadata_errors():
    error_message = 'Table metadata must be a dictionary.'
    with pytest.raises(TypeError, match=error_message):
        _check_table_metadata('test')


def test_check_column_metadata_errors(sample_column_names):
    error_message = 'Column metadata must be a dictionary.'
    with pytest.raises(TypeError, match=error_message):
        _check_column_metadata(sample_column_names, column_metadata='test')

    column_metadata = {
        'invalid_col': {'description': 'not a valid column'}
    }
    err_msg = re.escape("column_metadata contains columns that are not present in Schema: ['invalid_col']")
    with pytest.raises(LookupError, match=err_msg):
        _check_column_metadata(sample_column_names, column_metadata=column_metadata)


def test_check_column_description_errors(sample_column_names):
    error_message = 'column_descriptions must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_column_descriptions(sample_column_names, column_descriptions='test')

    column_descriptions = {
        'invalid_col': 'a description'
    }
    err_msg = re.escape("column_descriptions contains columns that are not present in Schema: ['invalid_col']")
    with pytest.raises(LookupError, match=err_msg):
        _check_column_descriptions(sample_column_names, column_descriptions=column_descriptions)


def test_schema_init(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types)

    assert schema.name is None
    assert schema.index is None
    assert schema.time_index is None

    assert set(schema.columns.keys()) == set(sample_column_names)


def test_schema_init_with_name(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    name='schema')

    assert schema.name == 'schema'
    assert schema.index is None
    assert schema.time_index is None


def test_schema_init_with_name_and_indices(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    name='schema',
                    index='id',
                    time_index='signup_date')

    assert schema.name == 'schema'
    assert schema.index == 'id'
    assert schema.time_index == 'signup_date'
    assert schema.columns[schema.time_index]['logical_type'] == Datetime


def test_schema_with_numeric_time_index(sample_column_names, sample_inferred_logical_types):
    # Set a numeric time index on init
    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'signup_date': 'Integer'}},
                    time_index='signup_date')
    date_col = schema.columns['signup_date']
    assert schema.time_index == 'signup_date'
    assert date_col['logical_type'] == Integer
    assert date_col['semantic_tags'] == {'time_index', 'numeric'}

    # Specify logical type for time index on init
    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'signup_date': 'Double'}},
                    time_index='signup_date')
    date_col = schema.columns['signup_date']
    assert schema.time_index == 'signup_date'
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


def test_schema_init_with_logical_type_classes(sample_column_names, sample_inferred_logical_types):
    logical_types = {
        'full_name': NaturalLanguage,
        'age': Double
    }
    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **logical_types},
                    name='schema')

    full_logical_types = {'id': Integer,
                          'full_name': NaturalLanguage,
                          'email': NaturalLanguage,
                          'phone_number': NaturalLanguage,
                          'age': Double,
                          'signup_date': Datetime,
                          'is_registered': Boolean}
    assert schema.logical_types == full_logical_types


def test_schema_init_with_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        'id': 'custom_tag',
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    name='schema',
                    semantic_tags=semantic_tags,
                    use_standard_tags=False)

    id_semantic_tags = schema.columns['id']['semantic_tags']
    assert isinstance(id_semantic_tags, set)
    assert len(id_semantic_tags) == 1
    assert 'custom_tag' in id_semantic_tags


def test_schema_adds_standard_semantic_tags(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'id': 'Categorical'}},
                    name='schema',
                    )

    assert schema.semantic_tags['id'] == {'category'}
    assert schema.semantic_tags['age'] == {'numeric'}

    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'id': 'Categorical'}},
                    name='schema',
                    use_standard_tags=False)

    assert schema.semantic_tags['id'] == set()
    assert schema.semantic_tags['age'] == set()


def test_semantic_tags_during_init(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'phone_number': ['tag3'],
        'signup_date': ['secondary_time_index'],
        'age': ['numeric', 'age']
    }
    expected_types = {
        'full_name': {'tag1'},
        'email': {'tag2'},
        'phone_number': {'tag3'},
        'signup_date': {'secondary_time_index'},
        'age': {'numeric', 'age'}
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types, semantic_tags=semantic_tags)
    assert schema.columns['full_name']['semantic_tags'] == expected_types['full_name']
    assert schema.columns['email']['semantic_tags'] == expected_types['email']
    assert schema.columns['phone_number']['semantic_tags'] == expected_types['phone_number']
    assert schema.columns['signup_date']['semantic_tags'] == expected_types['signup_date']
    assert schema.columns['age']['semantic_tags'] == expected_types['age']


def test_semantic_tag_errors(sample_column_names, sample_inferred_logical_types):
    error_message = "semantic_tags for column id must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        Schema(sample_column_names, sample_inferred_logical_types, semantic_tags={'id': int})

    error_message = "semantic_tags for column id must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        Schema(sample_column_names, sample_inferred_logical_types, semantic_tags={'id': {'index': {}, 'time_index': {}}})

    error_message = "semantic_tags for column id must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        Schema(sample_column_names, sample_inferred_logical_types, semantic_tags={'id': ['index', 1]})


def test_index_replacing_standard_tags(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types)
    assert schema.columns['id']['semantic_tags'] == {'numeric'}

    schema = Schema(sample_column_names, sample_inferred_logical_types, index='id')
    assert schema.columns['id']['semantic_tags'] == {'index'}


def test_raises_error_setting_index_tag_directly(sample_column_names, sample_inferred_logical_types):
    error_msg = re.escape("Cannot add 'index' tag directly. To set a column as the index, "
                          "use Schema.set_index() instead.")
    with pytest.raises(ValueError, match=error_msg):
        Schema(sample_column_names, sample_inferred_logical_types, semantic_tags={'id': 'index'})

    # --> add back when schema updates are implemented
    # Schema = Schema(sample_df)
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.add_semantic_tags({'id': 'index'})
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.set_semantic_tags({'id': 'index'})


def test_raises_error_setting_time_index_tag_directly(sample_column_names, sample_inferred_logical_types):
    error_msg = re.escape("Cannot add 'time_index' tag directly. To set a column as the time index, "
                          "use Schema.set_time_index() instead.")
    with pytest.raises(ValueError, match=error_msg):
        Schema(sample_column_names, sample_inferred_logical_types, semantic_tags={'signup_date': 'time_index'})

    # --> add back when schema updates are implemented
    # schema = Schema(sample_series)
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.add_semantic_tags({'signup_date': 'time_index'})
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.set_semantic_tags({'signup_date': 'time_index'})


def test_schema_init_with_col_descriptions(sample_column_names, sample_inferred_logical_types):
    descriptions = {
        'age': 'age of the user',
        'signup_date': 'date of account creation'
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types, column_descriptions=descriptions)
    for name, column in schema.columns.items():
        assert column['description'] == descriptions.get(name)


def test_schema_col_descriptions_errors(sample_column_names, sample_inferred_logical_types):
    # Errors at the table level
    err_msg = 'column_descriptions must be a dictionary'
    with pytest.raises(TypeError, match=err_msg):
        Schema(sample_column_names, sample_inferred_logical_types, column_descriptions=34)

    descriptions = {
        'invalid_col': 'not a valid column',
        'signup_date': 'date of account creation'
    }
    err_msg = re.escape("column_descriptions contains columns that are not present in Schema: ['invalid_col']")
    with pytest.raises(LookupError, match=err_msg):
        Schema(sample_column_names, sample_inferred_logical_types, column_descriptions=descriptions)

    # Errors at the column level
    descriptions = {
        'age': 7,
        'signup_date': 'date of account creation'
    }
    err_msg = "Column description must be a string"
    with pytest.raises(TypeError, match=err_msg):
        Schema(sample_column_names, sample_inferred_logical_types, column_descriptions=descriptions)


def test_schema_init_with_column_metadata(sample_column_names, sample_inferred_logical_types):
    column_metadata = {
        'age': {'interesting_values': [33]},
        'signup_date': {'description': 'date of account creation'}
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types, column_metadata=column_metadata)
    for name, column in schema.columns.items():
        assert column['metadata'] == (column_metadata.get(name) or {})


def test_column_metadata_errors(sample_column_names, sample_inferred_logical_types):
    column_metadata = {
        'age': 7,
        'signup_date': {'description': 'date of account creation'}
    }
    err_msg = "Column metadata must be a dictionary"
    with pytest.raises(TypeError, match=err_msg):
        Schema(sample_column_names, sample_inferred_logical_types, column_metadata=column_metadata)


# def test_numeric_time_index_dtypes(numeric_time_index_df):
#     schema = Schema(numeric_time_index_df, time_index='ints')
#     date_col = schema.columns['ints']
#     assert schema.time_index == 'ints'
#     assert date_col['logical_type'] == Integer
#     assert date_col['semantic_tags'] == {'time_index', 'numeric'}

#     # --> add back when schema updates are implemented
#     # schema = schema.set_time_index('floats')
#     # date_col = schema['floats']
#     # assert schema.time_index == 'floats'
#     # assert date_col.logical_type == Double
#     # assert date_col.semantic_tags == {'time_index', 'numeric'}

#     # schema = schema.set_time_index('with_null')
#     # date_col = schema['with_null']
#     # assert schema.time_index == 'with_null'
#     # if ks and isinstance(numeric_time_index_df, ks.DataFrame):
#     #     ltype = Double
#     # else:
#     #     ltype = Integer
#     # assert date_col.logical_type == ltype
#     # assert date_col.semantic_tags == {'time_index', 'numeric'}


# def test_numeric_index_strings(time_index_df):
#     error_msg = 'Time index column must contain datetime or numeric values'
#     with pytest.raises(TypeError, match=error_msg):
#         Schema(time_index_df, time_index='strs')

#     error_msg = 'Time index column must contain datetime or numeric values'
#     with pytest.raises(TypeError, match=error_msg):
#         Schema(time_index_df, time_index='ints', logical_types={'ints': 'Categorical'})

#     error_msg = 'Time index column must contain datetime or numeric values'
#     with pytest.raises(TypeError, match=error_msg):
#         Schema(time_index_df, time_index='letters', logical_types={'strs': 'Integer'})

#     schema = Schema(time_index_df, time_index='strs', logical_types={'strs': 'Double'})
#     date_col = schema.columns['strs']
#     assert schema.time_index == 'strs'
#     assert date_col['logical_type'] == Double
#     assert date_col['semantic_tags'] == {'time_index', 'numeric'}

#     # --> add back when schema updates are implemented
#     # schema = Schema(time_index_df, logical_types={'strs': 'Double'})
#     # schema = schema.set_time_index('strs')
#     # date_col = schema['strs']
#     # assert schema.time_index == 'strs'
#     # assert date_col.logical_type == Double
#     # assert date_col.semantic_tags == {'time_index', 'numeric'}


def test_ordinal_requires_instance_on_init(sample_column_names, sample_inferred_logical_types):
    error_msg = 'Must use an Ordinal instance with order values defined'
    with pytest.raises(TypeError, match=error_msg):
        Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'full_name': 'Ordinal'}})
    with pytest.raises(TypeError, match=error_msg):
        Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'full_name': Ordinal}})


def test_invalid_logical_type(sample_column_names, sample_inferred_logical_types):
    error_message = "Invalid logical type specified for 'full_name'"
    with pytest.raises(TypeError, match=error_message):
        Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'full_name': int}})

    error_message = "String naturalllanguage is not a valid logical type"
    with pytest.raises(ValueError, match=error_message):
        Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'full_name': 'naturalllanguage'}})
