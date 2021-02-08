import re

import pytest

import woodwork as ww
from woodwork.logical_types import Integer, Ordinal
from woodwork.schema_column import (
    _get_column_dict,
    _validate_description,
    _validate_logical_type,
    _validate_metadata
)


def test_validate_logical_type_errors():
    match = re.escape("logical_type <class 'int'> is not a registered LogicalType.")
    with pytest.raises(TypeError, match=match):
        _validate_logical_type(int)

    match = 'logical_type None is not a registered LogicalType.'
    with pytest.raises(TypeError, match=match):
        _validate_logical_type(None)

    ww.type_system.remove_type(Integer)
    match = 'logical_type Integer is not a registered LogicalType.'
    with pytest.raises(TypeError, match=match):
        _validate_logical_type(Integer)
    ww.type_system.reset_defaults()

    error_msg = 'Must use an Ordinal instance with order values defined'
    with pytest.raises(TypeError, match=error_msg):
        _get_column_dict('column', Ordinal)


def test_validate_description_errors():
    match = re.escape("Column description must be a string")
    with pytest.raises(TypeError, match=match):
        _validate_description(int)


def test_validate_metadata_errors():
    match = re.escape("Column metadata must be a dictionary")
    with pytest.raises(TypeError, match=match):
        _validate_metadata(int)


def test_get_column_dict():
    column = _get_column_dict('column', Integer, semantic_tags='test_tag')

    assert set(column.keys()) == {'name', 'dtype', 'logical_type', 'semantic_tags', 'description', 'metadata'}

    assert column.get('name') == 'column'
    assert column.get('logical_type') == Integer
    assert column.get('dtype') == 'Int64'
    assert column.get('semantic_tags') == {'numeric', 'test_tag'}

    assert column.get('description') is None
    assert column.get('metadata') == {}


def test_get_column_dict_standard_tags():
    column = _get_column_dict('column', Integer, use_standard_tags=False)

    assert column.get('semantic_tags') == set()


def test_get_column_dict_params():
    column = _get_column_dict('column', Integer, description='this is a column!', metadata={'created_by': 'user1'})

    assert column.get('description') == 'this is a column!'
    assert column.get('metadata') == {'created_by': 'user1'}


def test_raises_error_setting_index_tag_directly():
    error_msg = re.escape("Cannot add 'index' tag directly. To set a column as the index, "
                          "use Schema.set_index() instead.")
    with pytest.raises(ValueError, match=error_msg):
        _get_column_dict('column', Integer, semantic_tags='index')

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
        _get_column_dict('column', Integer, semantic_tags='time_index')

    # --> add back when schema updates are implemented
    # schema = Schema(sample_series)
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.add_semantic_tags({'signup_date': 'time_index'})
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.set_semantic_tags({'signup_date': 'time_index'})
