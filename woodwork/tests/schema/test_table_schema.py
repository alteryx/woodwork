import inspect
import re
from unittest.mock import patch

import pandas as pd
import pytest

from woodwork import type_system
from woodwork.exceptions import ColumnNotPresentError, DuplicateTagsWarning
from woodwork.logical_types import (
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    Integer,
    IntegerNullable,
    PersonFullName,
    PhoneNumber,
    Unknown,
)
from woodwork.table_schema import TableSchema


def test_schema_logical_types(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(sample_column_names, sample_inferred_logical_types)
    assert isinstance(schema.logical_types, dict)
    assert set(schema.logical_types.keys()) == set(sample_column_names)
    for k, v in schema.logical_types.items():
        assert v == schema.columns[k].logical_type


def test_schema_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {"full_name": "tag1", "email": ["tag2"], "age": ["numeric", "age"]}
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
    )
    assert isinstance(schema.semantic_tags, dict)
    assert set(schema.semantic_tags.keys()) == set(sample_column_names)
    for k, v in schema.semantic_tags.items():
        assert isinstance(v, set)
        assert v == schema.columns[k].semantic_tags


def test_schema_types(sample_column_names, sample_inferred_logical_types):
    sample_column_names.append("formatted_date")

    ymd_format = Datetime(datetime_format="%Y~%m~%d")
    schema = TableSchema(
        sample_column_names,
        logical_types={**sample_inferred_logical_types, "formatted_date": ymd_format},
        use_standard_tags=True,
    )

    returned_types = schema.types
    assert isinstance(returned_types, pd.DataFrame)
    assert "Logical Type" in returned_types.columns
    assert "Semantic Tag(s)" in returned_types.columns
    assert returned_types.shape[1] == 2
    assert len(returned_types.index) == len(sample_column_names)
    correct_logical_types = {
        name: ltype() for name, ltype in sample_inferred_logical_types.items()
    }
    correct_logical_types["formatted_date"] = ymd_format
    correct_logical_types = pd.Series(
        list(correct_logical_types.values()),
        index=list(correct_logical_types.keys()),
    )
    assert correct_logical_types.equals(returned_types["Logical Type"])

    correct_semantic_tags = {
        "id": "['numeric']",
        "full_name": "[]",
        "email": "[]",
        "phone_number": "[]",
        "age": "['numeric']",
        "signup_date": "[]",
        "is_registered": "[]",
        "double": "['numeric']",
        "double_with_nan": "['numeric']",
        "integer": "['numeric']",
        "nullable_integer": "['numeric']",
        "boolean": "[]",
        "categorical": "['category']",
        "datetime_with_NaT": "[]",
        "url": "[]",
        "ip_address": "[]",
        "formatted_date": "[]",
    }
    correct_semantic_tags = pd.Series(
        list(correct_semantic_tags.values()),
        index=list(correct_semantic_tags.keys()),
    )
    assert correct_semantic_tags.equals(returned_types["Semantic Tag(s)"])


def test_schema_repr(small_df):
    schema = TableSchema(
        list(small_df.columns),
        logical_types={"sample_datetime_series": Datetime},
    )

    schema_repr = repr(schema)
    expected_repr = "                       Logical Type Semantic Tag(s)\nColumn                                             \nsample_datetime_series     Datetime              []"
    assert schema_repr == expected_repr

    schema_html_repr = schema._repr_html_()
    expected_repr = '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>sample_datetime_series</th>\n      <td>Datetime</td>\n      <td>[]</td>\n    </tr>\n  </tbody>\n</table>'
    assert schema_html_repr == expected_repr


def test_schema_repr_empty():
    schema = TableSchema([], {})
    assert (
        repr(schema)
        == "Empty DataFrame\nColumns: [Logical Type, Semantic Tag(s)]\nIndex: []"
    )

    assert (
        schema._repr_html_()
        == '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n  </tbody>\n</table>'
    )


def test_schema_equality(sample_column_names, sample_inferred_logical_types):
    schema_basic = TableSchema(sample_column_names, sample_inferred_logical_types)
    schema_basic2 = TableSchema(sample_column_names, sample_inferred_logical_types)
    schema_names = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        name="test",
    )

    assert schema_basic != schema_names
    assert schema_basic == schema_basic2

    missing_col_names = sample_column_names[1:]
    missing_logical_types = sample_inferred_logical_types.copy()
    missing_logical_types.pop("id")

    schema_missing_col = TableSchema(missing_col_names, missing_logical_types)
    assert schema_basic != schema_missing_col

    schema_index = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
    )
    schema_time_index = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
    )

    assert schema_basic != schema_index
    assert schema_index != schema_time_index

    schema_numeric_time_index = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="id",
    )

    assert schema_time_index != schema_numeric_time_index

    schema_with_ltypes = TableSchema(
        sample_column_names,
        logical_types={**sample_inferred_logical_types, "full_name": Categorical},
        time_index="signup_date",
    )
    assert schema_with_ltypes != schema_time_index

    schema_with_metadata = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        table_metadata={"created_by": "user1"},
    )
    assert (
        TableSchema(sample_column_names, sample_inferred_logical_types, index="id")
        != schema_with_metadata
    )
    assert (
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            index="id",
            table_metadata={"created_by": "user1"},
        )
        == schema_with_metadata
    )
    assert (
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            index="id",
            table_metadata={"created_by": "user2"},
        )
        != schema_with_metadata
    )


def test_schema_equality_standard_tags(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags=True,
    )
    no_standard_tags_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags=False,
    )

    # Different standard tags values - using logical types with standard tags
    assert schema.use_standard_tags != no_standard_tags_schema.use_standard_tags
    assert schema != no_standard_tags_schema

    # Different standard tags values - no logical types that have standard tags
    unknown_types = {col_name: Unknown for col_name in sample_column_names}
    schema.set_types(logical_types=unknown_types)
    no_standard_tags_schema.set_types(logical_types=unknown_types)

    assert schema.use_standard_tags != no_standard_tags_schema.use_standard_tags
    assert schema != no_standard_tags_schema


def test_schema_shallow_equality(sample_column_names, sample_inferred_logical_types):
    metadata_table_1 = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        table_metadata={"user": "user0"},
    )
    metadata_table_2 = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        table_metadata={"user": "user0"},
    )

    assert metadata_table_1.__eq__(metadata_table_2, deep=False)
    assert metadata_table_1.__eq__(metadata_table_2, deep=True)

    diff_metadata_table = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        table_metadata={"user": "user1"},
    )

    assert metadata_table_1.__eq__(diff_metadata_table, deep=False)
    assert not metadata_table_1.__eq__(diff_metadata_table, deep=True)

    diff_col_metadata_table = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        table_metadata={"user": "user0"},
        column_metadata={"id": {"interesting_values": [0]}},
    )

    assert metadata_table_1.__eq__(diff_col_metadata_table, deep=False)
    assert not metadata_table_1.__eq__(diff_col_metadata_table, deep=True)

    diff_ltype_table = TableSchema(
        sample_column_names,
        {**sample_inferred_logical_types, "id": Categorical},
        table_metadata={"user": "user0"},
    )
    assert not metadata_table_1.__eq__(diff_ltype_table, deep=False)
    assert not metadata_table_1.__eq__(diff_ltype_table, deep=True)


def test_schema_table_metadata(sample_column_names, sample_inferred_logical_types):
    metadata = {
        "secondary_time_index": {"is_registered": "age"},
        "date_created": "11/13/20",
    }

    schema = TableSchema(sample_column_names, sample_inferred_logical_types)
    assert schema.metadata == {}

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        table_metadata=metadata,
        time_index="signup_date",
    )
    assert schema.metadata == metadata


def test_column_schema_metadata(sample_column_names, sample_inferred_logical_types):
    column_metadata = {"metadata_field": [1, 2, 3], "created_by": "user0"}

    schema = TableSchema(sample_column_names, sample_inferred_logical_types)
    assert schema.columns["id"].metadata == {}

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        column_metadata={"id": column_metadata},
    )
    assert schema.columns["id"].metadata == column_metadata


def test_filter_schema_cols_include(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
        index="id",
        name="df_name",
        use_standard_tags=True,
    )

    filtered = schema._filter_cols(include=Datetime)
    expected = {"signup_date", "datetime_with_NaT"}
    assert set(filtered) == expected

    filtered = schema._filter_cols(include="email", col_names=True)
    assert filtered == ["email"]

    filtered_log_type_string = schema._filter_cols(include="Unknown")
    filtered_log_type = schema._filter_cols(include=Unknown)
    expected = {"full_name"}
    assert filtered_log_type == filtered_log_type_string
    assert set(filtered_log_type) == expected
    expected = {"integer", "double", "double_with_nan", "age", "nullable_integer"}
    filtered_semantic_tag = schema._filter_cols(include="numeric")
    assert set(filtered_semantic_tag) == expected

    filtered_multiple_overlap = schema._filter_cols(
        include=["Unknown", "email"],
        col_names=True,
    )
    expected = ["full_name", "phone_number", "email"]
    for col in filtered_multiple_overlap:
        assert col in expected


def test_filter_schema_cols_exclude(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
        index="id",
        name="df_name",
        use_standard_tags=True,
    )

    filtered = schema._filter_cols(exclude=Datetime)
    assert "signup_date" not in filtered

    filtered = schema._filter_cols(exclude="email", col_names=True)
    assert "email" not in filtered

    filtered_log_type_string = schema._filter_cols(exclude="Unknown")
    filtered_log_type = schema._filter_cols(exclude=Unknown)
    expected = {
        "boolean",
        "double",
        "datetime_with_NaT",
        "email",
        "categorical",
        "double_with_nan",
        "signup_date",
        "age",
        "integer",
        "nullable_integer",
        "is_registered",
        "id",
        "url",
        "ip_address",
        "phone_number",
    }
    assert filtered_log_type == filtered_log_type_string
    assert set(filtered_log_type) == expected

    filtered_semantic_tag = schema._filter_cols(exclude="numeric")
    assert "age" not in filtered_semantic_tag

    filtered_multiple_overlap = schema._filter_cols(
        exclude=["Unknown", "email"],
        col_names=True,
    )
    expected = {
        "is_registered",
        "double",
        "nullable_integer",
        "double_with_nan",
        "categorical",
        "datetime_with_NaT",
        "age",
        "integer",
        "signup_date",
        "boolean",
        "id",
        "url",
        "ip_address",
        "phone_number",
    }
    assert set(filtered_multiple_overlap) == expected


def test_filter_schema_cols_no_matches(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
        index="id",
        name="df_name",
    )

    filter_no_matches = schema._filter_cols(include="nothing")
    assert filter_no_matches == []

    filter_empty_list = schema._filter_cols(include=[])
    assert filter_empty_list == []

    filter_non_string = schema._filter_cols(include=1)
    assert filter_non_string == []

    filter_exclude_no_matches = schema._filter_cols(exclude="nothing")
    assert set(filter_exclude_no_matches) == set(sample_column_names)

    filter_exclude_empty_list = schema._filter_cols(exclude=[])
    assert set(filter_exclude_empty_list) == set(sample_column_names)

    filter_exclude_non_string = schema._filter_cols(exclude=1)
    assert set(filter_exclude_non_string) == set(sample_column_names)


def test_filter_schema_errors(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
        index="id",
        name="df_name",
    )

    err_msg = "Invalid selector used in include: {} must be a string, uninstantiated and registered LogicalType, or valid column name"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(include=["boolean", "index", Double, {}])

    err_msg = "Invalid selector used in include: {} must be a string, uninstantiated and registered LogicalType, or valid column name"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(include=["boolean", "index", Double, {}], col_names=True)

    err_msg = "Invalid selector used in include: Datetime cannot be instantiated"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(Datetime())

    type_system.remove_type(EmailAddress)
    err_msg = "Specified LogicalType selector EmailAddress is not registered in Woodwork's type system."
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(EmailAddress)

    err_msg = "Invalid selector used in include: EmailAddress must be a string, uninstantiated and registered LogicalType, or valid column name"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(EmailAddress())
    type_system.reset_defaults()


def test_filter_schema_overlap_name_and_type(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(sample_column_names, sample_inferred_logical_types)

    filter_name_ltype_overlap = schema._filter_cols(include="full_name")
    assert filter_name_ltype_overlap == []

    filter_overlap_with_name = schema._filter_cols(include="full_name", col_names=True)
    assert filter_overlap_with_name == ["full_name"]

    schema = TableSchema(
        sample_column_names,
        {
            **sample_inferred_logical_types,
            "full_name": Categorical,
            "age": PersonFullName,
        },
        semantic_tags={"id": "person_full_name"},
    )

    filter_tag_and_ltype = schema._filter_cols(include="person_full_name")
    assert set(filter_tag_and_ltype) == {"id", "age"}

    filter_all_three = schema._filter_cols(
        include=["person_full_name", "full_name"],
        col_names=True,
    )
    assert set(filter_all_three) == {"id", "age", "full_name"}


def test_filter_schema_non_string_cols():
    schema = TableSchema(
        column_names=[0, 1, 2, 3],
        logical_types={0: Integer, 1: Categorical, 2: Unknown, 3: Double},
        use_standard_tags=True,
    )

    filter_types_and_tags = schema._filter_cols(include=[Integer, "category"])
    assert filter_types_and_tags == [0, 1]

    filter_by_name = schema._filter_cols(include=[0, 1], col_names=True)
    assert filter_by_name == [0, 1]


def test_get_subset_schema(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(sample_column_names, sample_inferred_logical_types)
    new_schema = schema.get_subset_schema(sample_column_names[1:4])
    for col in new_schema.columns:
        assert new_schema.semantic_tags[col] == schema.semantic_tags[col]
        assert new_schema.logical_types[col] == schema.logical_types[col]


def test_get_subset_schema_all_params(
    sample_column_names,
    sample_inferred_logical_types,
):
    # The first element is self, so it won't be included in kwargs
    possible_schema_params = inspect.getfullargspec(TableSchema.__init__)[0][1:]

    kwargs = {
        "column_names": sample_column_names,
        "logical_types": {**sample_inferred_logical_types, "email": EmailAddress},
        "name": "test_dt",
        "index": "id",
        "time_index": "signup_date",
        "semantic_tags": {"age": "test_tag"},
        "table_metadata": {"created_by": "user1"},
        "column_metadata": {"phone_number": {"format": "xxx-xxx-xxxx"}},
        "use_standard_tags": False,
        "column_descriptions": {"age": "this is a description"},
        "column_origins": "base",
        "validate": True,
    }

    # Confirm all possible params to TableSchema init are present with non-default values where possible
    assert set(possible_schema_params) == set(kwargs.keys())

    schema = TableSchema(**kwargs)
    copy_schema = schema.get_subset_schema(sample_column_names)

    assert schema == copy_schema
    assert schema is not copy_schema


def test_set_logical_types(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        "full_name": "tag1",
        "email": ["tag2"],
        "phone_number": ["tag3", "tag2"],
        "signup_date": {"secondary_time_index"},
    }
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=True,
    )

    schema.set_types(
        logical_types={
            "full_name": Categorical,
            "email": EmailAddress,
            "phone_number": PhoneNumber,
            "age": Double,
        },
    )

    assert schema.logical_types["full_name"] == Categorical
    assert schema.logical_types["email"] == EmailAddress
    assert schema.logical_types["phone_number"] == PhoneNumber
    assert schema.logical_types["age"] == Double

    # Verify custom tags were not removed after changing logical types
    assert schema.semantic_tags["full_name"] == {"category", "tag1"}
    assert schema.semantic_tags["email"] == {"tag2"}
    assert schema.semantic_tags["phone_number"] == {"tag3", "tag2"}
    assert schema.semantic_tags["age"] == {"numeric"}

    # Verify signup date column was unchanged
    assert isinstance(schema.logical_types["signup_date"], Datetime)
    assert schema.semantic_tags["signup_date"] == {"secondary_time_index"}


def test_set_logical_types_empty(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {"full_name": "tag1", "age": "test_tag"}
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="full_name",
        semantic_tags=semantic_tags,
        use_standard_tags=True,
    )

    # An empty set should reset the tags
    schema.set_types(semantic_tags={"full_name": set()}, retain_index_tags=False)
    assert isinstance(schema.logical_types["full_name"], Unknown)
    assert schema.semantic_tags["full_name"] == set()

    schema.set_types(semantic_tags={"age": set()})
    assert isinstance(schema.logical_types["age"], IntegerNullable)
    assert schema.semantic_tags["age"] == {"numeric"}


def test_set_logical_types_invalid_data(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(sample_column_names, sample_inferred_logical_types)

    error_message = re.escape(
        "logical_types contains columns that are not present in TableSchema: ['birthday']",
    )
    with pytest.raises(ColumnNotPresentError, match=error_message):
        schema.set_types(logical_types={"birthday": Double})

    error_message = (
        "Logical Types must be of the LogicalType class "
        "and registered in Woodwork's type system. "
        "Double does not meet that criteria."
    )
    with pytest.raises(TypeError, match=error_message):
        schema.set_types(logical_types={"id": "Double"})

    error_message = (
        "Logical Types must be of the LogicalType class "
        "and registered in Woodwork's type system. "
        "<class 'int'> does not meet that criteria."
    )
    with pytest.raises(TypeError, match=error_message):
        schema.set_types(logical_types={"age": int})

    error_message = "semantic_tags for full_name must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        schema.set_types(semantic_tags={"full_name": None})


def test_set_types_combined(sample_column_names, sample_inferred_logical_types):
    # test that the resetting of indices when ltype changes doesnt touch index??
    semantic_tags = {
        "id": "tag1",
        "email": ["tag2"],
        "phone_number": ["tag3", "tag2"],
        "signup_date": {"secondary_time_index"},
    }
    # use standard tags and keep index tags
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=True,
        index="id",
    )

    schema.set_types(
        semantic_tags={"id": "new_tag", "age": "new_tag", "email": "new_tag"},
        logical_types={"id": Double, "age": Double, "email": Categorical},
    )

    assert schema.semantic_tags["id"] == {"new_tag", "index"}
    assert schema.semantic_tags["age"] == {"numeric", "new_tag"}
    assert schema.semantic_tags["email"] == {"new_tag", "category"}

    assert schema.logical_types["id"] == Double
    assert schema.logical_types["age"] == Double
    assert schema.logical_types["email"] == Categorical

    # use standard tags and lose index tags
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=True,
        index="id",
        time_index="age",
    )

    schema.set_types(
        semantic_tags={"age": "new_tag"},
        logical_types={"id": Double},
        retain_index_tags=False,
    )

    assert schema.semantic_tags["id"] == {"numeric", "tag1"}
    assert schema.semantic_tags["age"] == {"numeric", "new_tag"}

    # don't use standard tags and lose index tags
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=False,
        index="id",
        time_index="signup_date",
    )

    schema.set_types(
        semantic_tags={"id": "new_tag", "age": "new_tag", "is_registered": "new_tag"},
        logical_types={"id": Double, "age": Double, "email": Categorical},
        retain_index_tags=False,
    )

    assert schema.index is None
    assert schema.time_index == "signup_date"

    assert schema.semantic_tags["id"] == {"new_tag"}
    assert schema.semantic_tags["age"] == {"new_tag"}
    assert schema.semantic_tags["is_registered"] == {"new_tag"}
    assert schema.semantic_tags["email"] == {"tag2"}
    assert schema.semantic_tags["signup_date"] == {"time_index", "secondary_time_index"}

    assert schema.logical_types["id"] == Double
    assert schema.logical_types["age"] == Double
    assert schema.logical_types["email"] == Categorical


def test_set_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {"full_name": "tag1", "age": ["numeric", "age"]}
    expected_tags = {"full_name": {"tag1"}, "age": {"numeric", "age"}}
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
    )
    assert schema.semantic_tags["full_name"] == expected_tags["full_name"]
    assert schema.semantic_tags["age"] == expected_tags["age"]

    new_tags = {
        "full_name": ["new_tag"],
        "age": "numeric",
    }
    schema.set_types(semantic_tags=new_tags)

    assert schema.semantic_tags["full_name"] == {"new_tag"}
    assert schema.semantic_tags["age"] == {"numeric"}


def test_set_semantic_tags_with_index(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        use_standard_tags=False,
    )
    assert schema.semantic_tags["id"] == {"index"}

    new_tags = {
        "id": "new_tag",
    }
    schema.set_types(semantic_tags=new_tags)
    assert schema.semantic_tags["id"] == {"index", "new_tag"}

    schema.set_types(semantic_tags=new_tags, retain_index_tags=False)
    assert schema.semantic_tags["id"] == {"new_tag"}


def test_set_semantic_tags_with_time_index(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
        use_standard_tags=False,
    )
    assert schema.semantic_tags["signup_date"] == {"time_index"}

    new_tags = {
        "signup_date": "new_tag",
    }
    schema.set_types(semantic_tags=new_tags)
    assert schema.semantic_tags["signup_date"] == {"time_index", "new_tag"}

    schema.set_types(semantic_tags=new_tags, retain_index_tags=False)
    assert schema.semantic_tags["signup_date"] == {"new_tag"}


def test_add_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {"full_name": "tag1", "age": ["numeric", "age"]}
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=False,
        index="id",
    )

    new_tags = {"full_name": ["list_tag"], "age": "str_tag", "id": {"set_tag"}}
    schema.add_semantic_tags(new_tags)

    assert schema.semantic_tags["full_name"] == {"tag1", "list_tag"}
    assert schema.semantic_tags["age"] == {"numeric", "age", "str_tag"}
    assert schema.semantic_tags["id"] == {"set_tag", "index"}


def test_warns_on_adding_duplicate_tags(
    sample_column_names,
    sample_inferred_logical_types,
):
    semantic_tags = {"full_name": "tag1", "age": ["numeric", "age"]}
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=False,
        index="id",
    )

    new_tags = {
        "full_name": "tag1",
    }
    expected_message = "Semantic tag(s) 'tag1' already present on column 'full_name'"
    with pytest.warns(DuplicateTagsWarning) as record:
        schema.add_semantic_tags(new_tags)
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message

    assert schema.semantic_tags["full_name"] == {"tag1"}


def test_reset_all_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {"full_name": "tag1", "age": "age"}
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=True,
    )

    schema.reset_semantic_tags()
    assert schema.semantic_tags["full_name"] == set()
    assert schema.semantic_tags["age"] == {"numeric"}


def test_reset_semantic_tags_with_index(
    sample_column_names,
    sample_inferred_logical_types,
):
    semantic_tags = {
        "id": "tag1",
    }
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        semantic_tags=semantic_tags,
        use_standard_tags=False,
    )
    assert schema.semantic_tags["id"] == {"index", "tag1"}

    schema.reset_semantic_tags("id", retain_index_tags=True)
    assert schema.semantic_tags["id"] == {"index"}

    schema.reset_semantic_tags("id")
    assert schema.semantic_tags["id"] == set()


def test_reset_semantic_tags_with_time_index(
    sample_column_names,
    sample_inferred_logical_types,
):
    semantic_tags = {
        "signup_date": "tag1",
    }
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
        semantic_tags=semantic_tags,
        use_standard_tags=False,
    )
    assert schema.semantic_tags["signup_date"] == {"time_index", "tag1"}

    schema.reset_semantic_tags("signup_date", retain_index_tags=True)
    assert schema.semantic_tags["signup_date"] == {"time_index"}

    schema.reset_semantic_tags("signup_date")
    assert schema.semantic_tags["signup_date"] == set()


def test_reset_semantic_tags_invalid_column(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
    )
    error_msg = re.escape("Column(s) '['invalid_column']' not found in DataFrame")
    with pytest.raises(ColumnNotPresentError, match=error_msg):
        schema.reset_semantic_tags("invalid_column")


def test_remove_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        "full_name": ["tag1", "tag2", "tag3"],
        "age": ["numeric", "age"],
        "id": ["tag1", "tag2"],
    }
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=False,
    )
    tags_to_remove = {"full_name": ["tag1", "tag3"], "age": "numeric", "id": {"tag1"}}
    schema.remove_semantic_tags(tags_to_remove)
    assert schema.semantic_tags["full_name"] == {"tag2"}
    assert schema.semantic_tags["age"] == {"age"}
    assert schema.semantic_tags["id"] == {"tag2"}


def test_raises_error_setting_index_tag_directly(
    sample_column_names,
    sample_inferred_logical_types,
):
    error_msg = re.escape(
        "Cannot add 'index' tag directly for column id. To set a column as the index, "
        "use DataFrame.ww.set_index() instead.",
    )

    schema = TableSchema(sample_column_names, sample_inferred_logical_types)

    with pytest.raises(ValueError, match=error_msg):
        schema.add_semantic_tags({"id": "index"})
    with pytest.raises(ValueError, match=error_msg):
        schema.set_types(semantic_tags={"id": "index"})


def test_raises_error_setting_time_index_tag_directly(
    sample_column_names,
    sample_inferred_logical_types,
):
    error_msg = re.escape(
        "Cannot add 'time_index' tag directly for column signup_date. To set a column as the time index, "
        "use DataFrame.ww.set_time_index() instead.",
    )
    schema = TableSchema(sample_column_names, sample_inferred_logical_types)

    with pytest.raises(ValueError, match=error_msg):
        schema.add_semantic_tags({"signup_date": "time_index"})
    with pytest.raises(ValueError, match=error_msg):
        schema.set_types(semantic_tags={"signup_date": "time_index"})


def test_removes_index_via_tags(sample_column_names, sample_inferred_logical_types):
    # Check setting tags
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        use_standard_tags=True,
    )
    schema.set_types(semantic_tags={"id": "new_tag"}, retain_index_tags=False)
    assert schema.semantic_tags["id"] == {"numeric", "new_tag"}
    assert schema.index is None

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        use_standard_tags=False,
    )
    schema.set_types(semantic_tags={"id": "new_tag"}, retain_index_tags=False)
    assert schema.semantic_tags["id"] == {"new_tag"}
    assert schema.index is None

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="full_name",
        use_standard_tags=True,
    )
    schema.set_types(semantic_tags={"full_name": "new_tag"}, retain_index_tags=False)
    assert schema.semantic_tags["full_name"] == {"new_tag"}
    assert schema.index is None

    # Check removing tags
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        use_standard_tags=True,
    )
    schema.remove_semantic_tags(semantic_tags={"id": "index"})
    assert schema.semantic_tags["id"] == {"numeric"}
    assert schema.index is None

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        use_standard_tags=False,
    )
    schema.remove_semantic_tags(semantic_tags={"id": "index"})
    assert schema.semantic_tags["id"] == set()
    assert schema.index is None

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="full_name",
        use_standard_tags=True,
    )
    schema.remove_semantic_tags(semantic_tags={"full_name": "index"})
    assert schema.semantic_tags["full_name"] == set()
    assert schema.index is None

    # Check resetting tags
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        use_standard_tags=True,
    )
    schema.reset_semantic_tags("id")
    assert schema.semantic_tags["id"] == {"numeric"}
    assert schema.index is None

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        use_standard_tags=False,
    )
    schema.reset_semantic_tags("id")
    assert schema.semantic_tags["id"] == set()
    assert schema.index is None

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="full_name",
        use_standard_tags=True,
    )
    schema.reset_semantic_tags("full_name")
    assert schema.semantic_tags["full_name"] == set()
    assert schema.index is None


def test_removes_time_index_via_tags(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
    )
    schema.set_types(semantic_tags={"signup_date": "new_tag"}, retain_index_tags=False)
    assert schema.semantic_tags["signup_date"] == {"new_tag"}
    assert schema.time_index is None

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
    )
    schema.remove_semantic_tags(semantic_tags={"signup_date": "time_index"})
    assert schema.semantic_tags["signup_date"] == set()
    assert schema.time_index is None

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        time_index="signup_date",
    )
    schema.reset_semantic_tags("signup_date")
    assert schema.semantic_tags["signup_date"] == set()
    assert schema.time_index is None


def test_set_index(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags=True,
    )
    assert schema.index is None
    assert schema.semantic_tags["id"] == {"numeric"}
    assert schema.semantic_tags["age"] == {"numeric"}

    schema.set_index("id")
    assert schema.index == "id"
    assert schema.semantic_tags["id"] == {"index"}

    schema.set_index("age")
    assert schema.index == "age"
    assert schema.semantic_tags["age"] == {"index"}
    assert schema.semantic_tags["id"] == {"numeric"}

    schema.set_index(None)
    assert schema.index is None
    assert schema.semantic_tags["age"] == {"numeric"}
    assert schema.semantic_tags["id"] == {"numeric"}


def test_set_index_errors(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(sample_column_names, sample_inferred_logical_types)

    error = re.escape("Specified index column `testing` not found in TableSchema.")
    with pytest.raises(ColumnNotPresentError, match=error):
        schema.set_index("testing")

    match = '"id" is already set as the index. '
    match += "An index cannot also be the time index."
    with pytest.raises(ValueError, match=match):
        TableSchema(
            sample_column_names,
            sample_inferred_logical_types,
            index="id",
            time_index="id",
        )

    schema.set_index("id")
    with pytest.raises(ValueError, match=match):
        schema.set_time_index("id")

    schema.set_time_index("signup_date")
    match = '"signup_date" is already set as the time index. '
    match += "A time index cannot also be the index."
    with pytest.raises(ValueError, match=match):
        schema.set_index("signup_date")


def test_set_time_index(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        use_standard_tags=True,
    )
    assert schema.time_index is None
    assert schema.semantic_tags["age"] == {"numeric"}
    assert schema.semantic_tags["signup_date"] == set()

    schema.set_time_index("signup_date")
    assert schema.time_index == "signup_date"
    assert schema.semantic_tags["signup_date"] == {"time_index"}

    schema.set_time_index("age")
    assert schema.time_index == "age"
    assert schema.semantic_tags["age"] == {"numeric", "time_index"}
    assert schema.semantic_tags["signup_date"] == set()

    schema.set_time_index(None)
    assert schema.index is None
    assert schema.semantic_tags["age"] == {"numeric"}
    assert schema.semantic_tags["signup_date"] == set()


def test_set_numeric_datetime_time_index(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        {**sample_inferred_logical_types, "age": Datetime},
    )

    assert isinstance(schema.logical_types["age"], Datetime)
    assert schema.semantic_tags["age"] == set()

    schema.set_time_index("age")
    assert schema.time_index == "age"
    assert schema.semantic_tags["age"] == {"time_index"}

    schema.set_time_index(None)
    assert schema.time_index is None
    assert schema.semantic_tags["age"] == set()


def test_set_time_index_errors(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(sample_column_names, sample_inferred_logical_types)

    error = re.escape("Specified time index column `testing` not found in TableSchema")
    with pytest.raises(ColumnNotPresentError, match=error):
        schema.set_time_index("testing")

    error = re.escape("Time index column must be a Datetime or numeric column.")
    with pytest.raises(TypeError, match=error):
        schema.set_time_index("email")


def test_set_index_twice(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(sample_column_names, sample_inferred_logical_types)
    schema.set_index(None)
    assert schema.index is None

    schema.set_time_index(None)
    assert schema.time_index is None

    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        time_index="signup_date",
    )
    original_schema = schema.get_subset_schema(list(schema.columns.keys()))

    schema.set_index("id")
    assert schema.index == "id"
    assert schema.semantic_tags["id"] == {"index"}
    assert schema == original_schema

    schema.set_time_index("signup_date")
    assert schema.time_index == "signup_date"
    assert schema.semantic_tags["signup_date"] == {"time_index"}
    assert schema == original_schema


def test_schema_rename_errors(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        time_index="signup_date",
    )

    error = "columns must be a dictionary"
    with pytest.raises(TypeError, match=error):
        schema.rename(1)

    error = "New columns names must be unique from one another."
    with pytest.raises(ValueError, match=error):
        schema.rename({"age": "test", "full_name": "test"})

    error = "Column to rename must be present. not_present cannot be found."
    with pytest.raises(ColumnNotPresentError, match=error):
        schema.rename({"not_present": "test"})

    error = "The column email is already present. Please choose another name to rename age to or also rename age."
    with pytest.raises(ValueError, match=error):
        schema.rename({"age": "email"})


def test_schema_rename(sample_column_names, sample_inferred_logical_types):
    table_metadata = {"table_info": "this is text"}
    id_description = "the id of the row"
    id_origin = "base"
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        time_index="signup_date",
        table_metadata=table_metadata,
        column_descriptions={"id": id_description},
        column_origins={"id": id_origin},
    )
    original_schema = schema.get_subset_schema(list(schema.columns.keys()))

    renamed_schema = schema.rename({"age": "birthday"})

    # Confirm original schema hasn't changed
    assert schema == original_schema

    assert "age" not in renamed_schema.columns
    assert "birthday" in renamed_schema.columns

    # confirm that metadata and descriptions are there
    assert renamed_schema.metadata == table_metadata
    assert schema.columns["id"].description == id_description
    assert schema.columns["id"].origin == id_origin

    old_col = schema.columns["age"]
    new_col = renamed_schema.columns["birthday"]
    assert old_col.logical_type == new_col.logical_type
    assert old_col.semantic_tags == new_col.semantic_tags

    swapped_schema = schema.rename({"age": "full_name", "full_name": "age"})
    swapped_back_schema = swapped_schema.rename(
        {"age": "full_name", "full_name": "age"},
    )
    assert swapped_back_schema == schema


def test_schema_rename_indices(sample_column_names, sample_inferred_logical_types):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
        index="id",
        time_index="signup_date",
    )

    renamed_schema = schema.rename(
        {"id": "renamed_index", "signup_date": "renamed_time_index"},
    )
    assert "id" not in renamed_schema.columns
    assert "signup_date" not in renamed_schema.columns
    assert "renamed_index" in renamed_schema.columns
    assert "renamed_time_index" in renamed_schema.columns

    assert renamed_schema.index == "renamed_index"
    assert renamed_schema.time_index == "renamed_time_index"


@patch("woodwork.table_schema._check_time_index")
@patch("woodwork.table_schema._check_index")
def test_validation_methods_called(
    mock_check_index,
    mock_check_time_index,
    sample_column_names,
    sample_inferred_logical_types,
):
    validation_schema = TableSchema(sample_column_names, sample_inferred_logical_types)
    no_validation_schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
    )

    assert not mock_check_index.called
    assert not mock_check_time_index.called

    no_validation_schema.set_index("id", validate=False)
    assert not mock_check_index.called

    validation_schema.set_index("id", validate=True)
    assert mock_check_index.called
    assert validation_schema == no_validation_schema

    validation_schema.set_time_index("signup_date", validate=False)
    assert not mock_check_time_index.called

    no_validation_schema.set_time_index("signup_date", validate=True)
    assert mock_check_time_index.called
    assert validation_schema == no_validation_schema


def test_schema_rename_preserves_order(
    sample_column_names,
    sample_inferred_logical_types,
):
    schema = TableSchema(
        sample_column_names,
        sample_inferred_logical_types,
    )
    rename_dict = {"id": "renamed_index", "signup_date": "renamed_time_index"}
    renamed_schema = schema.rename(rename_dict)
    expected_result = [
        rename_dict.get(col_name, col_name) for col_name in sample_column_names
    ]
    assert list(renamed_schema.columns.keys()) == expected_result
