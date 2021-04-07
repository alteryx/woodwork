import re

import pytest
from mock import patch

import woodwork as ww
from woodwork.column_schema import (
    ColumnSchema,
    _is_col_boolean,
    _is_col_categorical,
    _is_col_datetime,
    _is_col_numeric,
    _reset_semantic_tags,
    _validate_description,
    _validate_logical_type,
    _validate_metadata
)
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
    Ordinal
)


def test_validate_logical_type_errors():
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
        ColumnSchema('column', Ordinal)


def test_validate_description_errors():
    match = re.escape("Column description must be a string")
    with pytest.raises(TypeError, match=match):
        _validate_description(int)


def test_validate_metadata_errors():
    match = re.escape("Column metadata must be a dictionary")
    with pytest.raises(TypeError, match=match):
        _validate_metadata(int)


@patch("woodwork.column_schema._validate_metadata")
@patch("woodwork.column_schema._validate_description")
@patch("woodwork.column_schema._validate_logical_type")
def test_validation_methods_called(mock_validate_logical_type, mock_validate_description, mock_validate_metadata,
                                   sample_column_names, sample_inferred_logical_types):
    assert not mock_validate_logical_type.called
    assert not mock_validate_description.called
    assert not mock_validate_metadata.called

    not_validated_column = ColumnSchema('not_validated', logical_type=Integer,
                                        description='this is a description', metadata={'user': 'person1'},
                                        validate=False)
    assert not mock_validate_logical_type.called
    assert not mock_validate_description.called
    assert not mock_validate_metadata.called

    validated_column = ColumnSchema('not_validated', logical_type=Integer,
                                    description='this is a description', metadata={'user': 'person1'},
                                    validate=True)
    assert mock_validate_logical_type.called
    assert mock_validate_description.called
    assert mock_validate_metadata.called

    assert validated_column == not_validated_column


def test_column_schema():
    column = ColumnSchema('column', Integer, semantic_tags='test_tag')

    assert column.logical_type == Integer
    assert column.semantic_tags == {'test_tag'}

    assert column.description is None
    assert column.metadata == {}


def test_column_schema_standard_tags():
    column = ColumnSchema('column', Integer, use_standard_tags=True)

    assert column.semantic_tags == {'numeric'}


def test_column_schema_params():
    column = ColumnSchema('column', Integer, description='this is a column!', metadata={'created_by': 'user1'})

    assert column.description == 'this is a column!'
    assert column.metadata == {'created_by': 'user1'}


def test_column_schema_null_params():
    empty_col = ColumnSchema()
    assert empty_col.logical_type is None
    assert empty_col.description is None
    assert empty_col.semantic_tags == set()
    assert empty_col.metadata == {}

    just_tags = ColumnSchema(semantic_tags={'numeric', 'time_index'})
    assert just_tags.logical_type is None
    assert just_tags.semantic_tags == {'numeric', 'time_index'}

    just_ltype = ColumnSchema(logical_type=Integer)
    assert just_ltype.logical_type == Integer
    assert just_ltype.semantic_tags == set()

    error = "Cannot use standard tags when logical_type is None"
    with pytest.raises(ValueError, match=error):
        ColumnSchema(semantic_tags='categorical', use_standard_tags=True)

    error = "semantic_tags must be a string, set or list"
    with pytest.raises(TypeError, match=error):
        ColumnSchema(semantic_tags=1)


def test_is_col_numeric():
    int_column = ColumnSchema('ints', Integer)
    assert _is_col_numeric(int_column)

    double_column = ColumnSchema('floats', Double)
    assert _is_col_numeric(double_column)

    nl_column = ColumnSchema('text', NaturalLanguage)
    assert not _is_col_numeric(nl_column)

    manually_added = ColumnSchema('text', NaturalLanguage, semantic_tags='numeric')
    assert not _is_col_numeric(manually_added)

    no_standard_tags = ColumnSchema('ints', Integer, use_standard_tags=False)
    assert _is_col_numeric(no_standard_tags)

    instantiated_column = ColumnSchema('ints', Integer())
    assert _is_col_numeric(instantiated_column)


def test_is_col_categorical():
    categorical_column = ColumnSchema('cats', Categorical)
    assert _is_col_categorical(categorical_column)

    ordinal_column = ColumnSchema('ordinal', Ordinal(order=['a', 'b']))
    assert _is_col_categorical(ordinal_column)

    nl_column = ColumnSchema('text', NaturalLanguage)
    assert not _is_col_categorical(nl_column)

    manually_added = ColumnSchema('text', NaturalLanguage, semantic_tags='category')
    assert not _is_col_categorical(manually_added)

    no_standard_tags = ColumnSchema('cats', Categorical, use_standard_tags=False)
    assert _is_col_categorical(no_standard_tags)


def test_is_col_boolean():
    boolean_column = ColumnSchema('bools', Boolean)
    assert _is_col_boolean(boolean_column)

    instantiated_column = ColumnSchema('bools', Boolean())
    assert _is_col_boolean(instantiated_column)

    ordinal_column = ColumnSchema('ordinal', Ordinal(order=['a', 'b']))
    assert not _is_col_boolean(ordinal_column)

    nl_column = ColumnSchema('text', NaturalLanguage)
    assert not _is_col_boolean(nl_column)


def test_is_col_datetime():
    datetime_column = ColumnSchema('dates', Datetime)
    assert _is_col_datetime(datetime_column)

    formatted_datetime_column = ColumnSchema('dates', Datetime(datetime_format='%Y-%m%d'))
    assert _is_col_datetime(formatted_datetime_column)

    instantiated_datetime_column = ColumnSchema('dates', Datetime())
    assert _is_col_datetime(instantiated_datetime_column)

    nl_column = ColumnSchema('text', NaturalLanguage)
    assert not _is_col_datetime(nl_column)

    double_column = ColumnSchema('floats', Double)
    assert not _is_col_datetime(double_column)


def test_reset_semantic_tags_returns_new_object():
    standard_tags = {'tag1', 'tag2'}
    reset_tags = _reset_semantic_tags(standard_tags, use_standard_tags=True)
    assert reset_tags is not standard_tags
    assert reset_tags == standard_tags
