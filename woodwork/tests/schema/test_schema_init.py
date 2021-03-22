import re

import pytest

from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage
)
from woodwork.schema import (
    Schema,
    _check_column_descriptions,
    _check_column_metadata,
    _check_column_names,
    _check_index,
    _check_logical_types,
    _check_semantic_tags,
    _check_table_metadata,
    _check_time_index,
    _validate_params
)


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

    error_msg = 'Time index column must be a Datetime or numeric column'
    with pytest.raises(TypeError, match=error_msg):
        _check_time_index(column_names=sample_column_names, time_index='full_name', logical_type=NaturalLanguage)


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

    bad_logical_types_keys = {'email': 1}
    error_message = ("Logical Types must be of the LogicalType class "
                     "and registered in Woodwork's type system. "
                     "1 does not meet that criteria.")

    with pytest.raises(TypeError, match=error_message):
        _check_logical_types(sample_column_names, bad_logical_types_keys, require_all_cols=False)

    bad_logical_types_keys = {
        'email': 'NaturalLanguage',
    }
    error_message = ("Logical Types must be of the LogicalType class "
                     "and registered in Woodwork's type system. "
                     "NaturalLanguage does not meet that criteria.")

    with pytest.raises(TypeError, match=error_message):
        _check_logical_types(sample_column_names, bad_logical_types_keys, require_all_cols=False)


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
    error_message = re.escape("semantic_tags contains columns that do not exist: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_semantic_tags(sample_column_names, bad_semantic_tags_keys)

    error_message = "semantic_tags for id must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _check_semantic_tags(sample_column_names, {'id': 1})


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
    err_msg = re.escape("column_metadata contains columns that do not exist: ['invalid_col']")
    with pytest.raises(LookupError, match=err_msg):
        _check_column_metadata(sample_column_names, column_metadata=column_metadata)


def test_check_column_description_errors(sample_column_names):
    error_message = 'column_descriptions must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_column_descriptions(sample_column_names, column_descriptions='test')

    column_descriptions = {
        'invalid_col': 'a description'
    }
    err_msg = re.escape("column_descriptions contains columns that do not exist: ['invalid_col']")
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
    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'signup_date': Integer}},
                    time_index='signup_date')
    date_col = schema.columns['signup_date']
    assert schema.time_index == 'signup_date'
    assert date_col['logical_type'] == Integer
    assert date_col['semantic_tags'] == {'time_index', 'numeric'}

    # Specify logical type for time index on init
    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'signup_date': Double}},
                    time_index='signup_date')
    date_col = schema.columns['signup_date']
    assert schema.time_index == 'signup_date'
    assert date_col['logical_type'] == Double
    assert date_col['semantic_tags'] == {'time_index', 'numeric'}


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


def test_raises_error_setting_index_tag_directly(sample_column_names, sample_inferred_logical_types):
    error_msg = re.escape("Cannot add 'index' tag directly for column id. To set a column as the index, "
                          "use DataFrame.ww.set_index() instead.")
    with pytest.raises(ValueError, match=error_msg):
        semantic_tags = {'id': 'index'}
        Schema(sample_column_names, sample_inferred_logical_types,
               name='schema',
               semantic_tags=semantic_tags,
               use_standard_tags=False)


def test_raises_error_setting_time_index_tag_directly(sample_column_names, sample_inferred_logical_types):
    error_msg = re.escape("Cannot add 'time_index' tag directly for column signup_date. To set a column as the time index, "
                          "use DataFrame.ww.set_time_index() instead.")
    with pytest.raises(ValueError, match=error_msg):
        semantic_tags = {'signup_date': 'time_index'}
        Schema(sample_column_names, sample_inferred_logical_types,
               name='schema',
               semantic_tags=semantic_tags,
               use_standard_tags=False)


def test_schema_init_with_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {'id': 'custom_tag'}
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    name='schema',
                    semantic_tags=semantic_tags,
                    use_standard_tags=False)

    id_semantic_tags = schema.columns['id']['semantic_tags']
    assert isinstance(id_semantic_tags, set)
    assert len(id_semantic_tags) == 1
    assert 'custom_tag' in id_semantic_tags


def test_schema_adds_standard_semantic_tags(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'id': Categorical}},
                    name='schema')

    assert schema.semantic_tags['id'] == {'category'}
    assert schema.semantic_tags['age'] == {'numeric'}

    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'id': Categorical}},
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
    error_message = "semantic_tags for id must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        Schema(sample_column_names, sample_inferred_logical_types, semantic_tags={'id': int})

    error_message = "semantic_tags for id must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        Schema(sample_column_names, sample_inferred_logical_types, semantic_tags={'id': {'index': {}, 'time_index': {}}})

    error_message = "semantic_tags for id must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        Schema(sample_column_names, sample_inferred_logical_types, semantic_tags={'id': ['index', 1]})


def test_index_replacing_standard_tags(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types)
    assert schema.columns['id']['semantic_tags'] == {'numeric'}

    schema = Schema(sample_column_names, sample_inferred_logical_types, index='id')
    assert schema.columns['id']['semantic_tags'] == {'index'}


def test_schema_init_with_col_descriptions(sample_column_names, sample_inferred_logical_types):
    descriptions = {
        'age': 'age of the user',
        'signup_date': 'date of account creation'
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types, column_descriptions=descriptions)
    for name, column in schema.columns.items():
        assert column['description'] == descriptions.get(name)


def test_schema_col_descriptions_errors(sample_column_names, sample_inferred_logical_types):
    err_msg = 'column_descriptions must be a dictionary'
    with pytest.raises(TypeError, match=err_msg):
        Schema(sample_column_names, sample_inferred_logical_types, column_descriptions=34)

    descriptions = {
        'invalid_col': 'not a valid column',
        'signup_date': 'date of account creation'
    }
    err_msg = re.escape("column_descriptions contains columns that do not exist: ['invalid_col']")
    with pytest.raises(LookupError, match=err_msg):
        Schema(sample_column_names, sample_inferred_logical_types, column_descriptions=descriptions)


def test_schema_init_with_column_metadata(sample_column_names, sample_inferred_logical_types):
    column_metadata = {
        'age': {'interesting_values': [33]},
        'signup_date': {'description': 'date of account creation'}
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types, column_metadata=column_metadata)
    for name, column in schema.columns.items():
        assert column['metadata'] == (column_metadata.get(name) or {})
