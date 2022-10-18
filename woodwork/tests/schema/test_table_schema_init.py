import re
from unittest.mock import patch

import pytest

from woodwork.exceptions import ColumnNotPresentError
from woodwork.logical_types import (
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
    Ordinal,
)
from woodwork.table_schema import (
    TableSchema,
    _check_column_descriptions,
    _check_column_metadata,
    _check_column_names,
    _check_column_origins,
    _check_index,
    _check_logical_types,
    _check_semantic_tags,
    _check_table_metadata,
    _check_time_index,
    _check_use_standard_tags,
    _validate_params,
)


def test_validate_params_errors(sample_column_names):
    error_message = "Table name must be a string"
    with pytest.raises(TypeError, match=error_message):
        _validate_params(
            column_names=sample_column_names,
            name=1,
            index=None,
            time_index=None,
            logical_types=None,
            table_metadata=None,
            column_metadata=None,
            semantic_tags=None,
            column_descriptions=None,
            column_origins=None,
            use_standard_tags=False,
        )


def test_check_index_errors(sample_column_names):
    error_message = "Specified index column `foo` not found in TableSchema."
    with pytest.raises(ColumnNotPresentError, match=error_message):
        _check_index(column_names=sample_column_names, index="foo")


def test_check_time_index_errors(sample_column_names):
    error_message = "Specified time index column `foo` not found in TableSchema"
    with pytest.raises(ColumnNotPresentError, match=error_message):
        _check_time_index(
            column_names=sample_column_names,
            time_index="foo",
            logical_type=Integer,
        )

    error_msg = "Time index column must be a Datetime or numeric column"
    with pytest.raises(TypeError, match=error_msg):
        _check_time_index(
            column_names=sample_column_names,
            time_index="full_name",
            logical_type=NaturalLanguage,
        )


def test_check_column_names(sample_column_names):
    error_message = "Column names must be a list or set"
    with pytest.raises(TypeError, match=error_message):
        _check_column_names(column_names=int)

    sample_column_names.append("id")
    with pytest.raises(
        IndexError,
        match="TableSchema cannot contain duplicate columns names",
    ):
        _check_column_names(sample_column_names)


def test_check_logical_types_errors(sample_column_names):
    error_message = "logical_types must be a dictionary"
    with pytest.raises(TypeError, match=error_message):
        _check_logical_types(sample_column_names, logical_types="type")

    bad_logical_types_keys = {
        "full_name": None,
        "age": None,
        "birthday": None,
        "occupation": None,
    }
    error_message = re.escape(
        "logical_types contains columns that are not present in TableSchema: ['birthday', 'occupation']",
    )
    with pytest.raises(ColumnNotPresentError, match=error_message):
        _check_logical_types(sample_column_names, bad_logical_types_keys)

    bad_logical_types_keys = {
        "id": None,
        "full_name": None,
        "email": None,
        "phone_number": None,
        "age": None,
    }
    error_message = re.escape(
        "logical_types is missing columns that are present in TableSchema: "
        "['boolean', 'categorical', 'datetime_with_NaT', 'double', 'double_with_nan', "
        "'integer', 'ip_address', 'is_registered', 'nullable_integer', 'signup_date', 'url']",
    )
    with pytest.raises(ColumnNotPresentError, match=error_message):
        _check_logical_types(sample_column_names, bad_logical_types_keys)

    bad_logical_types_keys = {"email": 1}
    error_message = (
        "Logical Types must be of the LogicalType class "
        "and registered in Woodwork's type system. "
        "1 does not meet that criteria."
    )

    with pytest.raises(TypeError, match=error_message):
        _check_logical_types(
            sample_column_names,
            bad_logical_types_keys,
            require_all_cols=False,
        )

    bad_logical_types_keys = {
        "email": "NaturalLanguage",
    }
    error_message = (
        "Logical Types must be of the LogicalType class "
        "and registered in Woodwork's type system. "
        "NaturalLanguage does not meet that criteria."
    )

    with pytest.raises(TypeError, match=error_message):
        _check_logical_types(
            sample_column_names,
            bad_logical_types_keys,
            require_all_cols=False,
        )


def test_check_semantic_tags_errors(sample_column_names):
    error_message = "semantic_tags must be a dictionary"
    with pytest.raises(TypeError, match=error_message):
        _check_semantic_tags(sample_column_names, semantic_tags="type")

    bad_semantic_tags_keys = {
        "full_name": None,
        "age": None,
        "birthday": None,
        "occupation": None,
    }
    error_message = re.escape(
        "semantic_tags contains columns that are not present in TableSchema: ['birthday', 'occupation']",
    )
    with pytest.raises(ColumnNotPresentError, match=error_message):
        _check_semantic_tags(sample_column_names, bad_semantic_tags_keys)

    error_message = "semantic_tags for id must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _check_semantic_tags(sample_column_names, {"id": 1})


def test_check_table_metadata_errors():
    error_message = "Table metadata must be a dictionary."
    with pytest.raises(TypeError, match=error_message):
        _check_table_metadata("test")


def test_check_column_metadata_errors(sample_column_names):
    error_message = "Column metadata must be a dictionary."
    with pytest.raises(TypeError, match=error_message):
        _check_column_metadata(sample_column_names, column_metadata="test")

    column_metadata = {"invalid_col": {"description": "not a valid column"}}
    err_msg = re.escape(
        "column_metadata contains columns that are not present in TableSchema: ['invalid_col']",
    )
    with pytest.raises(ColumnNotPresentError, match=err_msg):
        _check_column_metadata(sample_column_names, column_metadata=column_metadata)


def test_check_column_description_errors(sample_column_names):
    error_message = "column_descriptions must be a dictionary"
    with pytest.raises(TypeError, match=error_message):
        _check_column_descriptions(sample_column_names, column_descriptions="test")

    column_descriptions = {"invalid_col": "a description"}
    err_msg = re.escape(
        "column_descriptions contains columns that are not present in TableSchema: ['invalid_col']",
    )
    with pytest.raises(ColumnNotPresentError, match=err_msg):
        _check_column_descriptions(
            sample_column_names,
            column_descriptions=column_descriptions,
        )


def test_check_column_origin_errors(sample_column_names):
    error_message = "column_origins must be a dictionary or a string"
    with pytest.raises(TypeError, match=error_message):
        _check_column_origins(sample_column_names, column_origins=123)

    column_origins = {"invalid_col": "base"}
    err_msg = re.escape(
        "column_origins contains columns that are not present in TableSchema: ['invalid_col']",
    )
    with pytest.raises(ColumnNotPresentError, match=err_msg):
        _check_column_origins(sample_column_names, column_origins=column_origins)


def test_check_use_standard_tags_errors(sample_column_names):
    error_message = "use_standard_tags must be a dictionary or a boolean"
    with pytest.raises(TypeError, match=error_message):
        _check_use_standard_tags(sample_column_names, use_standard_tags=1)

    error_message = re.escape(
        "use_standard_tags contains columns that are not present in TableSchema: ['invalid_col']",
    )
    with pytest.raises(ColumnNotPresentError, match=error_message):
        _check_use_standard_tags(
            sample_column_names,
            use_standard_tags={"invalid_col": True},
        )

    error_message = "use_standard_tags for column id must be a boolean"
    with pytest.raises(TypeError, match=error_message):
        _check_use_standard_tags(sample_column_names, use_standard_tags={"id": 1})


def test_schema_init(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(sample_column_names, sample_inferred_logical_types)

    assert schema.name is None
    assert schema.index is None
    assert schema.time_index is None

    assert set(schema.columns.keys()) == set(sample_column_names)


def test_schema_init_with_name(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        name="schema",
    )

    assert schema.name == "schema"
    assert schema.index is None
    assert schema.time_index is None


def test_schema_init_with_name_and_indices(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        name="schema",
        index="id",
        time_index="signup_date",
    )

    assert schema.name == "schema"
    assert schema.index == "id"
    assert schema.time_index == "signup_date"
    assert isinstance(schema.columns[schema.time_index].logical_type, Datetime)


def test_schema_with_numeric_time_index(
    sample_column_names,
    sample_inferred_logical_types,
):
    # Set a numeric time index on init
    schema = TableSchema(
        sample_column_names,
        logical_types={**sample_inferred_logical_types, **{"signup_date": Integer}},
        time_index="signup_date",
        use_standard_tags=True,
    )
    date_col = schema.columns["signup_date"]
    assert schema.time_index == "signup_date"
    assert isinstance(date_col.logical_type, Integer)
    assert date_col.semantic_tags == {"time_index", "numeric"}

    # Specify logical type for time index on init
    schema = TableSchema(
        sample_column_names,
        logical_types={**sample_inferred_logical_types, **{"signup_date": Double}},
        time_index="signup_date",
        use_standard_tags=True,
    )
    date_col = schema.columns["signup_date"]
    assert schema.time_index == "signup_date"
    assert isinstance(date_col.logical_type, Double)
    assert date_col.semantic_tags == {"time_index", "numeric"}


def test_schema_init_with_logical_type_classes(
    sample_column_names,
    sample_correct_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        logical_types=sample_correct_logical_types,
        name="schema",
    )
    instantiated_ltypes = {
        name: ltype() for name, ltype in sample_correct_logical_types.items()
    }
    assert schema.logical_types == instantiated_ltypes


def test_raises_error_setting_index_tag_directly(
    sample_column_names,
    sample_inferred_logical_types,
):
    error_msg = re.escape(
        "Cannot add 'index' tag directly for column id. To set a column as the index, "
        "use DataFrame.ww.set_index() instead.",
    )
    with pytest.raises(ValueError, match=error_msg):
        semantic_tags = {"id": "index"}
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            name="schema",
            semantic_tags=semantic_tags,
            use_standard_tags=False,
        )


def test_raises_error_setting_time_index_tag_directly(
    sample_column_names,
    sample_inferred_logical_types,
):
    error_msg = re.escape(
        "Cannot add 'time_index' tag directly for column signup_date. To set a column as the time index, "
        "use DataFrame.ww.set_time_index() instead.",
    )
    with pytest.raises(ValueError, match=error_msg):
        semantic_tags = {"signup_date": "time_index"}
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            name="schema",
            semantic_tags=semantic_tags,
            use_standard_tags=False,
        )


def test_schema_init_with_semantic_tags(
    sample_column_names,
    sample_inferred_logical_types,
):
    semantic_tags = {"id": "custom_tag"}
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        name="schema",
        semantic_tags=semantic_tags,
        use_standard_tags=False,
    )

    id_semantic_tags = schema.columns["id"].semantic_tags
    assert isinstance(id_semantic_tags, set)
    assert len(id_semantic_tags) == 1
    assert "custom_tag" in id_semantic_tags


def test_schema_adds_standard_semantic_tags(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        logical_types={**sample_inferred_logical_types, **{"id": Categorical}},
        use_standard_tags=True,
        name="schema",
    )

    assert schema.semantic_tags["id"] == {"category"}
    assert schema.semantic_tags["age"] == {"numeric"}

    schema = TableSchema(
        sample_column_names,
        logical_types={**sample_inferred_logical_types, **{"id": Categorical}},
        name="schema",
        use_standard_tags=False,
    )

    assert schema.semantic_tags["id"] == set()
    assert schema.semantic_tags["age"] == set()


def test_semantic_tags_during_init(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        "full_name": "tag1",
        "email": ["tag2"],
        "phone_number": ["tag3"],
        "signup_date": ["secondary_time_index"],
        "age": ["numeric", "age"],
    }
    expected_types = {
        "full_name": {"tag1"},
        "email": {"tag2"},
        "phone_number": {"tag3"},
        "signup_date": {"secondary_time_index"},
        "age": {"numeric", "age"},
    }
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
    )
    assert schema.columns["full_name"].semantic_tags == expected_types["full_name"]
    assert schema.columns["email"].semantic_tags == expected_types["email"]
    assert (
        schema.columns["phone_number"].semantic_tags == expected_types["phone_number"]
    )
    assert schema.columns["signup_date"].semantic_tags == expected_types["signup_date"]
    assert schema.columns["age"].semantic_tags == expected_types["age"]


def test_semantic_tag_errors(sample_column_names, sample_inferred_logical_types):
    error_message = "semantic_tags for id must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            semantic_tags={"id": int},
        )

    error_message = "semantic_tags for id must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            semantic_tags={"id": {"index": {}, "time_index": {}}},
        )

    error_message = "semantic_tags for id must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            semantic_tags={"id": ["index", 1]},
        )


def test_index_replacing_standard_tags(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags=True,
    )
    assert schema.columns["id"].semantic_tags == {"numeric"}

    schema = TableSchema(sample_column_names, sample_inferred_logical_types, index="id")
    assert schema.columns["id"].semantic_tags == {"index"}


def test_schema_init_with_col_descriptions(
    sample_column_names,
    sample_inferred_logical_types,
):
    descriptions = {"age": "age of the user", "signup_date": "date of account creation"}
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        column_descriptions=descriptions,
    )
    for name, column in schema.columns.items():
        assert column.description == descriptions.get(name)


def test_schema_col_descriptions_errors(
    sample_column_names,
    sample_inferred_logical_types,
):
    err_msg = "column_descriptions must be a dictionary"
    with pytest.raises(TypeError, match=err_msg):
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            column_descriptions=34,
        )

    descriptions = {
        "invalid_col": "not a valid column",
        "signup_date": "date of account creation",
    }
    err_msg = re.escape(
        "column_descriptions contains columns that are not present in TableSchema: ['invalid_col']",
    )
    with pytest.raises(ColumnNotPresentError, match=err_msg):
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            column_descriptions=descriptions,
        )


def test_schema_init_with_col_origins(
    sample_column_names,
    sample_inferred_logical_types,
):
    origins = {"age": "base", "signup_date": "engineered"}
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        column_origins=origins,
    )
    for name, column in schema.columns.items():
        assert column.origin == origins.get(name)

    schema_single_origin = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        column_origins="base",
    )
    for name, column in schema_single_origin.columns.items():
        assert column.origin == "base"


def test_schema_col_origins_errors(sample_column_names, sample_inferred_logical_types):
    err_msg = "column_origins must be a dictionary or a string"
    with pytest.raises(TypeError, match=err_msg):
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            column_origins=34,
        )

    origins = {"invalid_col": "not a valid column", "signup_date": "base"}
    err_msg = re.escape(
        "column_origins contains columns that are not present in TableSchema: ['invalid_col']",
    )
    with pytest.raises(ColumnNotPresentError, match=err_msg):
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            column_origins=origins,
        )

    origins = {"signup_date": 1}
    err_msg = "Column origin must be a string"
    with pytest.raises(TypeError, match=err_msg):
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            column_origins=origins,
        )


def test_schema_init_with_column_metadata(
    sample_column_names,
    sample_inferred_logical_types,
):
    column_metadata = {
        "age": {"interesting_values": [33]},
        "signup_date": {"description": "date of account creation"},
    }
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        column_metadata=column_metadata,
    )
    for name, column in schema.columns.items():
        assert column.metadata == (column_metadata.get(name) or {})


@patch("woodwork.table_schema._validate_not_setting_index_tags")
@patch("woodwork.table_schema._check_time_index")
@patch("woodwork.table_schema._check_index")
@patch("woodwork.table_schema._validate_params")
def test_validation_methods_called(
    mock_validate_params,
    mock_check_index,
    mock_check_time_index,
    mock_validate_not_setting_index,
    sample_column_names,
    sample_inferred_logical_types,
):
    assert not mock_validate_params.called
    assert not mock_check_index.called
    assert not mock_check_time_index.called
    assert not mock_validate_not_setting_index.called

    not_validated_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        time_index="signup_date",
        validate=False,
    )
    assert not mock_validate_params.called
    assert not mock_check_index.called
    assert not mock_check_time_index.called
    assert not mock_validate_not_setting_index.called

    validated_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        time_index="signup_date",
        validate=True,
    )
    assert mock_validate_params.called
    assert mock_check_index.called
    assert mock_check_time_index.called
    assert mock_validate_not_setting_index.called

    assert validated_schema == not_validated_schema


def test_use_standard_tags_from_bool(
    sample_column_names,
    sample_inferred_logical_types,
):
    standard_tags_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags=True,
    )
    assert set(standard_tags_schema.columns.keys()) == set(
        standard_tags_schema.use_standard_tags.keys(),
    )
    assert all([*standard_tags_schema.use_standard_tags.values()])
    assert standard_tags_schema.semantic_tags["id"] == {"numeric"}

    no_standard_tags_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags=False,
    )
    assert set(no_standard_tags_schema.columns.keys()) == set(
        no_standard_tags_schema.use_standard_tags.keys(),
    )
    assert not any([*no_standard_tags_schema.use_standard_tags.values()])
    assert no_standard_tags_schema.semantic_tags["id"] == set()


def test_use_standard_tags_from_dict(
    sample_column_names,
    sample_inferred_logical_types,
):
    default_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags={col_name: False for col_name in sample_column_names},
    )
    assert default_schema.use_standard_tags == {
        col_name: False for col_name in sample_column_names
    }

    use_standard_tags = {
        "id": True,
        "full_name": False,
        "email": True,
        "phone_number": True,
        "age": False,
        "signup_date": True,
        "is_registered": False,
        "double": False,
        "double_with_nan": False,
        "integer": False,
        "nullable_integer": False,
        "boolean": False,
        "categorical": False,
        "datetime_with_NaT": False,
        "url": False,
        "ip_address": False,
    }
    full_dict_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags=use_standard_tags,
    )
    assert full_dict_schema.use_standard_tags == use_standard_tags

    partial_dict_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags={
            "id": True,
            "email": True,
            "phone_number": True,
            "signup_date": True,
        },
    )
    assert full_dict_schema.use_standard_tags == partial_dict_schema.use_standard_tags
    assert full_dict_schema == partial_dict_schema

    partial_dict_default_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags={
            "id": False,
            "email": False,
            "phone_number": False,
            "signup_date": False,
        },
    )
    assert (
        default_schema.use_standard_tags
        == partial_dict_default_schema.use_standard_tags
    )
    assert default_schema == partial_dict_default_schema


def test_ordinal_without_init():
    schema = TableSchema(
        column_names=["ordinal_col"],
        logical_types={"ordinal_col": Ordinal},
    )
    assert isinstance(schema.logical_types["ordinal_col"], Ordinal)
    assert schema.logical_types["ordinal_col"].order is None
