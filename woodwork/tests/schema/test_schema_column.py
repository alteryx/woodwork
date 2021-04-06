import re

import pytest
from mock import patch

import woodwork as ww
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
    Ordinal
)
from woodwork.schema_column import (
    _get_column_dict,
    _is_col_boolean,
    _is_col_categorical,
    _is_col_datetime,
    _is_col_numeric,
    _reset_semantic_tags,
    _validate_description,
    _validate_logical_type,
    _validate_metadata
)


def test_validate_logical_type_errors():
    assert _validate_logical_type(None) is None

    match = re.escape("logical_type <class 'int'> is not a registered LogicalType.")
    with pytest.raises(TypeError, match=match):
        _validate_logical_type(int)

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


@patch("woodwork.schema_column._validate_metadata")
@patch("woodwork.schema_column._validate_description")
@patch("woodwork.schema_column._validate_logical_type")
def test_validation_methods_called(mock_validate_logical_type, mock_validate_description, mock_validate_metadata,
                                   sample_column_names, sample_inferred_logical_types):
    assert not mock_validate_logical_type.called
    assert not mock_validate_description.called
    assert not mock_validate_metadata.called

    not_validated_column = _get_column_dict('not_validated', logical_type=Integer,
                                            description='this is a description', metadata={'user': 'person1'},
                                            validate=False)
    assert not mock_validate_logical_type.called
    assert not mock_validate_description.called
    assert not mock_validate_metadata.called

    validated_column = _get_column_dict('not_validated', logical_type=Integer,
                                        description='this is a description', metadata={'user': 'person1'},
                                        validate=True)
    assert mock_validate_logical_type.called
    assert mock_validate_description.called
    assert mock_validate_metadata.called

    assert validated_column == not_validated_column


def test_get_column_dict():
    column = _get_column_dict('column', Integer, semantic_tags='test_tag')

    assert set(column.keys()) == {'logical_type', 'semantic_tags', 'description', 'metadata'}

    assert column.get('logical_type') == Integer
    assert column.get('semantic_tags') == {'test_tag'}

    assert column.get('description') is None
    assert column.get('metadata') == {}


def test_get_column_dict_standard_tags():
    column = _get_column_dict('column', Integer, use_standard_tags=True)

    assert column.get('semantic_tags') == {'numeric'}


def test_get_column_dict_params():
    column = _get_column_dict('column', Integer, description='this is a column!', metadata={'created_by': 'user1'})

    assert column.get('description') == 'this is a column!'
    assert column.get('metadata') == {'created_by': 'user1'}


def test_get_column_dict_null_params():
    empty_col = {
        'logical_type': None,
        'semantic_tags': set(),
        'description': None,
        'metadata': {}
    }
    assert _get_column_dict() == empty_col

    just_tags = _get_column_dict(semantic_tags={'numeric', 'time_index'})
    assert just_tags.get('logical_type') is None
    assert just_tags.get('semantic_tags') == {'numeric', 'time_index'}

    just_tags = _get_column_dict(logical_type=Integer)
    assert just_tags.get('logical_type') == Integer
    assert just_tags.get('semantic_tags') == set()

    error = "Cannot use standard tags when logical_type is None"
    with pytest.raises(ValueError, match=error):
        _get_column_dict(semantic_tags='categorical', use_standard_tags=True)

    error = "semantic_tags for None must be a string, set or list"
    with pytest.raises(TypeError, match=error):
        _get_column_dict(semantic_tags=1)


def test_is_col_numeric():
    int_column = _get_column_dict('ints', Integer)
    assert _is_col_numeric(int_column)

    double_column = _get_column_dict('floats', Double)
    assert _is_col_numeric(double_column)

    nl_column = _get_column_dict('text', NaturalLanguage)
    assert not _is_col_numeric(nl_column)

    manually_added = _get_column_dict('text', NaturalLanguage, semantic_tags='numeric')
    assert not _is_col_numeric(manually_added)

    no_standard_tags = _get_column_dict('ints', Integer, use_standard_tags=False)
    assert _is_col_numeric(no_standard_tags)

    instantiated_column = _get_column_dict('ints', Integer())
    assert _is_col_numeric(instantiated_column)


def test_is_col_categorical():
    categorical_column = _get_column_dict('cats', Categorical)
    assert _is_col_categorical(categorical_column)

    ordinal_column = _get_column_dict('ordinal', Ordinal(order=['a', 'b']))
    assert _is_col_categorical(ordinal_column)

    nl_column = _get_column_dict('text', NaturalLanguage)
    assert not _is_col_categorical(nl_column)

    manually_added = _get_column_dict('text', NaturalLanguage, semantic_tags='category')
    assert not _is_col_categorical(manually_added)

    no_standard_tags = _get_column_dict('cats', Categorical, use_standard_tags=False)
    assert _is_col_categorical(no_standard_tags)


def test_is_col_boolean():
    boolean_column = _get_column_dict('bools', Boolean)
    assert _is_col_boolean(boolean_column)

    instantiated_column = _get_column_dict('bools', Boolean())
    assert _is_col_boolean(instantiated_column)

    ordinal_column = _get_column_dict('ordinal', Ordinal(order=['a', 'b']))
    assert not _is_col_boolean(ordinal_column)

    nl_column = _get_column_dict('text', NaturalLanguage)
    assert not _is_col_boolean(nl_column)


def test_is_col_datetime():
    datetime_column = _get_column_dict('dates', Datetime)
    assert _is_col_datetime(datetime_column)

    formatted_datetime_column = _get_column_dict('dates', Datetime(datetime_format='%Y-%m%d'))
    assert _is_col_datetime(formatted_datetime_column)

    instantiated_datetime_column = _get_column_dict('dates', Datetime())
    assert _is_col_datetime(instantiated_datetime_column)

    nl_column = _get_column_dict('text', NaturalLanguage)
    assert not _is_col_datetime(nl_column)

    double_column = _get_column_dict('floats', Double)
    assert not _is_col_datetime(double_column)


def test_reset_semantic_tags_returns_new_object():
    standard_tags = {'tag1', 'tag2'}
    reset_tags = _reset_semantic_tags(standard_tags, use_standard_tags=True)
    assert reset_tags is not standard_tags
    assert reset_tags == standard_tags
