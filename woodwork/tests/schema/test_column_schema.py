import re

import pytest
from mock import patch

import woodwork as ww
from woodwork.column_schema import (
    ColumnSchema,
    _validate_description,
    _validate_logical_type,
    _validate_metadata,
    _validate_origin
)
from woodwork.exceptions import (
    DuplicateTagsWarning,
    StandardTagsChangedWarning
)
from woodwork.logical_types import (
    Boolean,
    BooleanNullable,
    Categorical,
    Datetime,
    Double,
    Integer,
    IntegerNullable,
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


def test_validate_description_errors():
    match = re.escape("Column description must be a string")
    with pytest.raises(TypeError, match=match):
        _validate_description(int)


def test_validate_origin_errors():
    match = re.escape("Column origin must be a string")
    with pytest.raises(TypeError, match=match):
        _validate_origin(int)


def test_validate_metadata_errors():
    match = re.escape("Column metadata must be a dictionary")
    with pytest.raises(TypeError, match=match):
        _validate_metadata(int)


@patch("woodwork.column_schema._validate_metadata")
@patch("woodwork.column_schema._validate_description")
@patch("woodwork.column_schema._validate_origin")
@patch("woodwork.column_schema._validate_logical_type")
def test_validation_methods_called(mock_validate_logical_type, mock_validate_description, mock_validate_origin,
                                   mock_validate_metadata, sample_column_names, sample_inferred_logical_types):
    assert not mock_validate_logical_type.called
    assert not mock_validate_description.called
    assert not mock_validate_origin.called
    assert not mock_validate_metadata.called

    not_validated_column = ColumnSchema(logical_type=Integer,
                                        description='this is a description',
                                        origin='base', metadata={'user': 'person1'},
                                        validate=False)
    assert not mock_validate_logical_type.called
    assert not mock_validate_description.called
    assert not mock_validate_origin.called
    assert not mock_validate_metadata.called

    validated_column = ColumnSchema(logical_type=Integer,
                                    description='this is a description',
                                    origin='base', metadata={'user': 'person1'},
                                    validate=True)
    assert mock_validate_logical_type.called
    assert mock_validate_description.called
    assert mock_validate_origin.called
    assert mock_validate_metadata.called

    assert validated_column == not_validated_column


def test_column_schema():
    column = ColumnSchema(logical_type=Integer, semantic_tags='test_tag')

    assert isinstance(column.logical_type, Integer)
    assert column.semantic_tags == {'test_tag'}

    assert column.description is None
    assert column.origin is None
    assert column.metadata == {}


def test_column_schema_standard_tags():
    column = ColumnSchema(logical_type=Integer, use_standard_tags=True)

    assert column.semantic_tags == {'numeric'}


def test_column_schema_params():
    column = ColumnSchema(logical_type=Integer, description='this is a column!', origin='base',
                          metadata={'created_by': 'user1'})

    assert column.description == 'this is a column!'
    assert column.origin == 'base'
    assert column.metadata == {'created_by': 'user1'}


def test_column_schema_null_params():
    empty_col = ColumnSchema()
    assert empty_col.logical_type is None
    assert empty_col.description is None
    assert empty_col.origin is None
    assert empty_col.semantic_tags == set()
    assert empty_col.metadata == {}

    just_tags = ColumnSchema(semantic_tags={'numeric', 'time_index'})
    assert just_tags.logical_type is None
    assert just_tags.semantic_tags == {'numeric', 'time_index'}

    just_ltype = ColumnSchema(logical_type=Integer)
    assert isinstance(just_ltype.logical_type, Integer)
    assert just_ltype.semantic_tags == set()

    error = "Cannot use standard tags when logical_type is None"
    with pytest.raises(ValueError, match=error):
        ColumnSchema(semantic_tags='categorical', use_standard_tags=True)

    error = "semantic_tags must be a string, set or list"
    with pytest.raises(TypeError, match=error):
        ColumnSchema(semantic_tags=1)


def test_is_numeric():
    int_column = ColumnSchema(logical_type=Integer)
    assert int_column.is_numeric

    nullable_int_column = ColumnSchema(logical_type=IntegerNullable)
    assert nullable_int_column.is_numeric

    double_column = ColumnSchema(logical_type=Double)
    assert double_column.is_numeric

    nl_column = ColumnSchema(logical_type=NaturalLanguage)
    assert not nl_column.is_numeric

    manually_added = ColumnSchema(logical_type=NaturalLanguage, semantic_tags='numeric')
    assert not manually_added.is_numeric

    no_standard_tags = ColumnSchema(logical_type=Integer, use_standard_tags=False)
    assert no_standard_tags.is_numeric

    instantiated_column = ColumnSchema(logical_type=Integer())
    assert instantiated_column.is_numeric

    empty_column = ColumnSchema()
    assert not empty_column.is_numeric


def test_is_categorical():
    categorical_column = ColumnSchema(logical_type=Categorical)
    assert categorical_column.is_categorical

    ordinal_column = ColumnSchema(logical_type=Ordinal(order=['a', 'b']))
    assert ordinal_column.is_categorical

    nl_column = ColumnSchema(logical_type=NaturalLanguage)
    assert not nl_column.is_categorical

    manually_added = ColumnSchema(logical_type=NaturalLanguage, semantic_tags='category')
    assert not manually_added.is_categorical

    no_standard_tags = ColumnSchema(logical_type=Categorical, use_standard_tags=False)
    assert no_standard_tags.is_categorical

    empty_column = ColumnSchema()
    assert not empty_column.is_categorical


def test_is_boolean():
    boolean_column = ColumnSchema(logical_type=Boolean)
    assert boolean_column.is_boolean

    nullable_boolean_column = ColumnSchema(logical_type=BooleanNullable)
    assert nullable_boolean_column.is_boolean

    instantiated_column = ColumnSchema(logical_type=Boolean())
    assert instantiated_column.is_boolean

    ordinal_column = ColumnSchema(logical_type=Ordinal(order=['a', 'b']))
    assert not ordinal_column.is_boolean

    nl_column = ColumnSchema(logical_type=NaturalLanguage)
    assert not nl_column.is_boolean

    empty_column = ColumnSchema()
    assert not empty_column.is_boolean


def test_is_datetime():
    datetime_column = ColumnSchema(logical_type=Datetime)
    assert datetime_column.is_datetime

    formatted_datetime_column = ColumnSchema(logical_type=Datetime(datetime_format='%Y-%m%d'))
    assert formatted_datetime_column.is_datetime

    instantiated_datetime_column = ColumnSchema(logical_type=Datetime())
    assert instantiated_datetime_column.is_datetime

    nl_column = ColumnSchema(logical_type=NaturalLanguage)
    assert not nl_column.is_datetime

    double_column = ColumnSchema(logical_type=Double)
    assert not double_column.is_datetime

    empty_column = ColumnSchema()
    assert not empty_column.is_datetime


def test_set_semantic_tags_no_standard_tags():
    semantic_tags = {'tag1', 'tag2'}
    schema = ColumnSchema(logical_type=Categorical, semantic_tags=semantic_tags)
    assert schema.semantic_tags == semantic_tags

    new_tag = ['new_tag']
    schema._set_semantic_tags(new_tag)
    assert schema.semantic_tags == set(new_tag)


def test_set_semantic_tags_with_standard_tags():
    semantic_tags = {'tag1', 'tag2'}
    schema = ColumnSchema(logical_type=Categorical, semantic_tags=semantic_tags, use_standard_tags=True)
    assert schema.semantic_tags == semantic_tags.union({'category'})

    new_tag = ['new_tag']
    schema._set_semantic_tags(new_tag)
    assert schema.semantic_tags == set(new_tag).union({'category'})


def test_set_semantic_tags_no_logical_type():
    semantic_tags = {'tag1', 'tag2'}
    schema = ColumnSchema(semantic_tags=semantic_tags, use_standard_tags=False)

    new_tag = ['new_tag']
    schema._set_semantic_tags(new_tag)
    assert schema.semantic_tags == set(new_tag)


def test_add_custom_tags():
    semantic_tags = 'initial_tag'
    schema = ColumnSchema(logical_type=Categorical, semantic_tags=semantic_tags, use_standard_tags=True)

    schema._add_semantic_tags('string_tag', 'col_name')
    assert schema.semantic_tags == {'initial_tag', 'string_tag', 'category'}

    schema._add_semantic_tags(['list_tag'], 'col_name')
    assert schema.semantic_tags == {'initial_tag', 'string_tag', 'list_tag', 'category'}

    schema._add_semantic_tags({'set_tag'}, 'col_name')
    assert schema.semantic_tags == {'initial_tag', 'string_tag', 'list_tag', 'set_tag', 'category'}


def test_add_custom_tags_no_logical_type():
    semantic_tags = 'initial_tag'
    schema = ColumnSchema(semantic_tags=semantic_tags, use_standard_tags=False)

    schema._add_semantic_tags('string_tag', 'col_name')
    assert schema.semantic_tags == {'initial_tag', 'string_tag'}

    schema._add_semantic_tags(['list_tag'], 'col_name')
    assert schema.semantic_tags == {'initial_tag', 'string_tag', 'list_tag'}

    schema._add_semantic_tags({'set_tag'}, 'col_name')
    assert schema.semantic_tags == {'initial_tag', 'string_tag', 'list_tag', 'set_tag'}


def test_warns_on_adding_duplicate_tag():
    semantic_tags = ['first_tag', 'second_tag']
    schema = ColumnSchema(logical_type=Categorical, semantic_tags=semantic_tags, use_standard_tags=False)

    expected_message = "Semantic tag(s) 'first_tag, second_tag' already present on column 'col_name'"
    with pytest.warns(DuplicateTagsWarning) as record:
        schema._add_semantic_tags(['first_tag', 'second_tag'], 'col_name')
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message


def test_reset_semantic_tags_with_standard_tags():
    semantic_tags = 'initial_tag'
    schema = ColumnSchema(semantic_tags=semantic_tags,
                          logical_type=Categorical,
                          use_standard_tags=True)

    schema._reset_semantic_tags()
    assert schema.semantic_tags == Categorical.standard_tags


def test_reset_semantic_tags_without_standard_tags():
    semantic_tags = 'initial_tag'
    schema = ColumnSchema(semantic_tags=semantic_tags, use_standard_tags=False)

    schema._reset_semantic_tags()
    assert schema.semantic_tags == set()


def test_reset_semantic_tags_returns_new_object():
    schema = ColumnSchema(logical_type=Integer, semantic_tags=set(), use_standard_tags=True)
    standard_tags = Integer.standard_tags

    schema._reset_semantic_tags()
    assert schema.semantic_tags is not standard_tags
    assert schema.semantic_tags == standard_tags


def test_remove_semantic_tags():
    tags_to_remove = [
        'tag1',
        ['tag1'],
        {'tag1'}
    ]

    for tag in tags_to_remove:
        schema = ColumnSchema(semantic_tags=['tag1', 'tag2'], use_standard_tags=False)
        schema._remove_semantic_tags(tag, 'col_name')
        assert schema.semantic_tags == {'tag2'}


def test_remove_standard_semantic_tag():
    # Check that warning is raised if use_standard_tags is True - tag should be removed
    schema = ColumnSchema(logical_type=Categorical, semantic_tags='tag1', use_standard_tags=True)
    expected_message = 'Standard tags have been removed from "col_name"'
    with pytest.warns(StandardTagsChangedWarning) as record:
        schema._remove_semantic_tags(['tag1', 'category'], 'col_name')
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message
    assert schema.semantic_tags == set()

    # Check that warning is not raised if use_standard_tags is False - tag should be removed
    schema = ColumnSchema(logical_type=Categorical, semantic_tags=['category', 'tag1'], use_standard_tags=False)

    with pytest.warns(None) as record:
        schema._remove_semantic_tags(['tag1', 'category'], 'col_name')
    assert len(record) == 0
    assert schema.semantic_tags == set()

    # Check that warning is not raised if use_standard_tags is False and no Logical Type is specified
    schema = ColumnSchema(semantic_tags=['category', 'tag1'], use_standard_tags=False)

    with pytest.warns(None) as record:
        schema._remove_semantic_tags(['tag1', 'category'], 'col_name')
    assert len(record) == 0
    assert schema.semantic_tags == set()


def test_remove_semantic_tags_raises_error_with_invalid_tag():
    schema = ColumnSchema(logical_type=Categorical, semantic_tags='tag1')
    error_msg = re.escape("Semantic tag(s) 'invalid_tagname' not present on column 'col_name'")
    with pytest.raises(LookupError, match=error_msg):
        schema._remove_semantic_tags('invalid_tagname', 'col_name')


def test_schema_equality():
    col = ColumnSchema(logical_type=Categorical)
    diff_description_col = ColumnSchema(logical_type=Categorical, description='description')
    diff_origin_col = ColumnSchema(logical_type=Categorical, origin='base')
    diff_metadata_col = ColumnSchema(logical_type=Categorical, metadata={'interesting_values': ['a', 'b']})
    use_standard_tags_col = ColumnSchema(logical_type=Categorical, use_standard_tags=True)
    diff_tags_col = ColumnSchema(logical_type=Categorical, semantic_tags={'new_tag'})

    assert col != diff_description_col
    assert col != diff_origin_col
    assert col != diff_metadata_col
    assert col != use_standard_tags_col
    assert col != diff_tags_col

    # Check columns with same logical types but different parameters
    ordinal_ltype_1 = Ordinal(order=['a', 'b', 'c'])
    ordinal_ltype_2 = Ordinal(order=['b', 'a', 'c'])
    ordinal_col_1 = ColumnSchema(logical_type=ordinal_ltype_1)
    ordinal_col_2 = ColumnSchema(logical_type=ordinal_ltype_2)

    assert col != ordinal_col_1
    assert ordinal_col_1 != ordinal_col_2
    assert ordinal_col_1 == ordinal_col_1

    datetime_ltype_instantiated = Datetime(datetime_format='%Y-%m%d')

    datetime_col_format = ColumnSchema(logical_type=datetime_ltype_instantiated)
    datetime_col_param = ColumnSchema(logical_type=Datetime(datetime_format=None))
    datetime_col_instantiated = ColumnSchema(logical_type=Datetime())

    assert datetime_col_instantiated != datetime_col_format
    assert datetime_col_instantiated == datetime_col_param


def test_schema_shallow_equality():
    no_metadata_1 = ColumnSchema(logical_type=Categorical)
    no_metadata_2 = ColumnSchema(logical_type=Categorical)

    assert no_metadata_1.__eq__(no_metadata_2, deep=False)
    assert no_metadata_1.__eq__(no_metadata_2, deep=True)

    metadata_1 = ColumnSchema(logical_type=Categorical, metadata={'interesting_values': ['a', 'b']})
    metadata_2 = ColumnSchema(logical_type=Categorical, metadata={'interesting_values': ['a', 'b']})
    metadata_3 = ColumnSchema(logical_type=Categorical, metadata={'interesting_values': ['c', 'd']})

    assert metadata_1.__eq__(metadata_2, deep=False)
    assert metadata_1.__eq__(metadata_2, deep=True)
    assert metadata_1.__eq__(metadata_3, deep=False)
    assert not metadata_1.__eq__(metadata_3, deep=True)


def test_schema_repr():
    assert (repr(ColumnSchema(logical_type=Datetime, semantic_tags='time_index')) ==
            "<ColumnSchema (Logical Type = Datetime) (Semantic Tags = ['time_index'])>")
    assert (repr(ColumnSchema(logical_type=Integer)) ==
            "<ColumnSchema (Logical Type = Integer)>")
    assert (repr(ColumnSchema(semantic_tags={'category', 'foreign_key'})) ==
            "<ColumnSchema (Semantic Tags = ['category', 'foreign_key'])>")
    assert (repr(ColumnSchema()) ==
            "<ColumnSchema>")


def test_ordinal_without_init():
    schema = ColumnSchema(logical_type=Ordinal)
    assert isinstance(schema.logical_type, Ordinal)
    assert schema.logical_type.order is None
