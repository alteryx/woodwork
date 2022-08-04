from unittest.mock import patch

import pandas as pd
import pytest

import woodwork as ww
from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.logical_types import (
    BooleanNullable,
    Categorical,
    Double,
    Integer,
    IntegerNullable,
)
from woodwork.tests.testing_utils import to_pandas
from woodwork.utils import concat_columns


def test_concat_cols_ww_dfs(sample_df):
    sample_df.ww.init(
        logical_types={"full_name": "Categorical"},
        semantic_tags={"age": "test_tag"},
        table_metadata={"created_by": "user0"},
    )
    df1 = sample_df.ww[
        [
            "id",
            "full_name",
            "email",
            "phone_number",
            "age",
            "signup_date",
            "is_registered",
        ]
    ]
    df2 = sample_df.ww[
        [
            "double",
            "double_with_nan",
            "integer",
            "nullable_integer",
            "boolean",
            "categorical",
            "datetime_with_NaT",
            "url",
            "ip_address",
        ]
    ]
    df2.ww.metadata = None

    combined_df = concat_columns([df1, df2])
    assert combined_df.ww == sample_df.ww
    assert isinstance(combined_df.ww.logical_types["full_name"], Categorical)
    assert "test_tag" in combined_df.ww.semantic_tags["age"]
    assert combined_df.ww.metadata == {"created_by": "user0"}

    pandas_combined_df = pd.concat(
        [to_pandas(df1), to_pandas(df2)],
        axis=1,
        join="outer",
        ignore_index=False,
    )
    assert to_pandas(combined_df).equals(pandas_combined_df)


def test_concat_cols_uninit_dfs(sample_df):
    df1 = sample_df[
        [
            "id",
            "full_name",
            "email",
            "phone_number",
            "age",
            "signup_date",
            "is_registered",
        ]
    ]
    df2 = sample_df[
        [
            "double",
            "double_with_nan",
            "integer",
            "nullable_integer",
            "boolean",
            "categorical",
            "datetime_with_NaT",
            "url",
            "ip_address",
        ]
    ]
    sample_df.ww.init()

    combined_df = concat_columns([df1, df2])
    assert df1.ww.schema is None
    assert df2.ww.schema is None
    assert combined_df.ww.schema is not None

    # Though the input dataframes don't have Woodwork initalized,
    # the concatenated DataFrame has Woodwork initialized for all the columns
    assert isinstance(combined_df.ww.logical_types["id"], Integer)
    assert combined_df.ww == sample_df.ww

    df1.ww.init()
    df2.ww.init()
    pandas_combined_df = pd.concat(
        [to_pandas(df1), to_pandas(df2)],
        axis=1,
        join="outer",
        ignore_index=False,
    )
    assert to_pandas(combined_df).equals(pandas_combined_df)


def test_concat_cols_combo_dfs(sample_df):
    df1 = sample_df[
        [
            "id",
            "full_name",
            "email",
            "phone_number",
            "age",
            "signup_date",
            "is_registered",
        ]
    ]

    sample_df.ww.init()
    df2 = sample_df.ww[
        [
            "double",
            "double_with_nan",
            "integer",
            "nullable_integer",
            "boolean",
            "categorical",
            "datetime_with_NaT",
            "url",
            "ip_address",
        ]
    ]

    combined_df = concat_columns([df1, df2])
    assert df1.ww.schema is None
    assert combined_df.ww == sample_df.ww

    df1.ww.init()
    df2.ww.init()
    pandas_combined_df = pd.concat(
        [to_pandas(df1), to_pandas(df2)],
        axis=1,
        join="outer",
        ignore_index=False,
    )
    assert to_pandas(combined_df).equals(pandas_combined_df)


def test_concat_cols_with_series(sample_df):
    expected_df = sample_df[["id", "full_name", "signup_date", "is_registered"]]
    df = expected_df[["id", "full_name"]]
    s1 = expected_df["signup_date"]
    expected_df.ww.init()
    s2 = expected_df["is_registered"]

    combined_df = concat_columns([df, s1, s2])
    assert combined_df.ww == expected_df.ww

    df.ww.init()
    s1.ww.init()
    s2.ww.init()
    pandas_combined_df = pd.concat(
        [to_pandas(df), to_pandas(s1), to_pandas(s2)],
        axis=1,
        join="outer",
        ignore_index=False,
    )
    assert to_pandas(combined_df).equals(pandas_combined_df)


def test_concat_cols_with_conflicting_ww_indexes(sample_df):
    df1 = sample_df[
        [
            "id",
            "full_name",
            "email",
            "phone_number",
            "age",
            "signup_date",
            "is_registered",
        ]
    ]
    df1.ww.init(index="id")
    df2 = sample_df[
        [
            "double",
            "double_with_nan",
            "integer",
            "nullable_integer",
            "boolean",
            "categorical",
            "datetime_with_NaT",
        ]
    ]
    df2.ww.init(index="double")

    error = (
        "Cannot set the Woodwork index of multiple input objects. "
        "Please remove the index columns from all but one table."
    )

    with pytest.raises(IndexError, match=error):
        concat_columns([df1, df2])

    df1 = sample_df[
        ["id", "full_name", "email", "phone_number", "age", "is_registered"]
    ]
    df1.ww.init(time_index="id")
    df2 = sample_df[
        [
            "signup_date",
            "double",
            "double_with_nan",
            "integer",
            "nullable_integer",
            "boolean",
            "categorical",
            "datetime_with_NaT",
        ]
    ]
    df2.ww.init(time_index="signup_date")

    error = (
        "Cannot set the Woodwork time index of multiple input objects. "
        "Please remove the time index columns from all but one table."
    )
    with pytest.raises(IndexError, match=error):
        concat_columns([df1, df2])


def test_concat_cols_with_ww_indexes(sample_df):
    df1 = sample_df[
        ["id", "full_name", "email", "phone_number", "age", "is_registered"]
    ]
    df1.ww.init(index="id")
    df2 = sample_df[
        [
            "signup_date",
            "double",
            "double_with_nan",
            "integer",
            "nullable_integer",
            "boolean",
            "categorical",
            "datetime_with_NaT",
        ]
    ]
    df2.ww.init(time_index="signup_date")

    combined_df = concat_columns([df1, df2])
    assert combined_df.ww.index == "id"
    assert combined_df.ww.time_index == "signup_date"


def test_concat_cols_with_duplicate_ww_indexes(sample_df):
    df1 = sample_df[
        [
            "id",
            "signup_date",
            "full_name",
            "email",
            "phone_number",
            "age",
            "is_registered",
        ]
    ]
    df1.ww.init(index="id", time_index="signup_date")
    df2 = sample_df[
        [
            "id",
            "signup_date",
            "double",
            "double_with_nan",
            "integer",
            "nullable_integer",
            "boolean",
            "categorical",
            "datetime_with_NaT",
            "url",
            "ip_address",
        ]
    ]
    df2.ww.init(index="id", time_index="signup_date")

    error = (
        "Cannot set the Woodwork index of multiple input objects. "
        "Please remove the index columns from all but one table."
    )
    with pytest.raises(IndexError, match=error):
        concat_columns([df1, df2])

    df2.ww.pop("id")

    error = (
        "Cannot set the Woodwork time index of multiple input objects. "
        "Please remove the time index columns from all but one table."
    )
    with pytest.raises(IndexError, match=error):
        concat_columns([df1, df2])

    df2.ww.pop("signup_date")

    # Because underlying index is set, this won't change concat operation
    pd.testing.assert_index_equal(to_pandas(df1.index), to_pandas(df2.index))

    combined_df = concat_columns([df1, df2])
    assert combined_df.ww.index == "id"
    assert combined_df.ww.time_index == "signup_date"
    assert len(combined_df.columns) == len(sample_df.columns)


def test_concat_table_names(sample_df):
    df1 = sample_df[["id", "signup_date"]]
    df1.ww.init(name=0)
    df2 = sample_df[["full_name", "age"]]
    df2.ww.init(name="1")
    df3 = sample_df[["phone_number", "is_registered"]]
    df3.ww.init()

    combined_df = concat_columns([df1, df2, df3])
    assert combined_df.ww.name == "0_1"

    combined_df = concat_columns([df3, df2, df1])
    assert combined_df.ww.name == "1_0"


def test_concat_cols_different_use_standard_tags(sample_df):
    sample_df.ww.init(
        use_standard_tags={"age": False, "full_name": False, "id": True},
        logical_types={"full_name": "Categorical"},
    )

    assert sample_df.ww.semantic_tags["id"] == {"numeric"}
    assert sample_df.ww.semantic_tags["full_name"] == set()
    assert sample_df.ww.semantic_tags["age"] == set()

    df1 = sample_df.ww[["id", "full_name", "email"]]
    df2 = sample_df.ww[["phone_number", "age", "signup_date", "is_registered"]]
    combined_df = concat_columns([df1, df2])

    assert combined_df.ww.semantic_tags["id"] == {"numeric"}
    assert combined_df.ww.semantic_tags["full_name"] == set()
    assert combined_df.ww.semantic_tags["age"] == set()
    assert isinstance(combined_df.ww.logical_types["full_name"], Categorical)
    assert not combined_df.ww.use_standard_tags["age"]
    assert not combined_df.ww.use_standard_tags["full_name"]
    assert combined_df.ww.use_standard_tags["id"]


def test_concat_combine_metadatas(sample_df):
    df1 = sample_df[["id", "full_name", "email"]]
    df1.ww.init(
        table_metadata={"created_by": "user0", "test_key": "test_val1"},
        column_metadata={"id": {"interesting_values": [1, 2]}},
    )
    df2 = sample_df[["phone_number", "age", "signup_date", "is_registered"]]
    df2.ww.init(
        table_metadata={"table_type": "single", "test_key": "test_val2"},
        column_metadata={"age": {"interesting_values": [33]}},
    )

    error = "Cannot resolve overlapping keys in table metadata: {'test_key'}"
    with pytest.raises(ValueError, match=error):
        concat_columns([df1, df2])

    del df2.ww.metadata["test_key"]

    combined_df = concat_columns([df1, df2])

    assert combined_df.ww.metadata == {
        "created_by": "user0",
        "table_type": "single",
        "test_key": "test_val1",
    }
    assert combined_df.ww.columns["id"].metadata == {"interesting_values": [1, 2]}
    assert combined_df.ww.columns["age"].metadata == {"interesting_values": [33]}


@patch("woodwork.table_accessor._validate_accessor_params")
def test_concat_cols_validate_schema(mock_validate_accessor_params, sample_df):
    df1 = sample_df[["id", "full_name", "email"]]
    df1.ww.init(validate=False)
    df2 = sample_df[["phone_number", "age", "signup_date", "is_registered"]]
    df2.ww.init(validate=False)

    assert not mock_validate_accessor_params.called

    concat_columns([df1, df2], validate_schema=False)

    assert not mock_validate_accessor_params.called

    concat_columns([df1, df2], validate_schema=True)

    assert mock_validate_accessor_params.called


def test_concat_cols_mismatched_index_adds_single_nan(sample_df):
    # If the dtype can handle nans, it won't change
    sample_df.ww.init(logical_types={"id": "IntegerNullable"})

    df1 = sample_df.ww.loc[[0, 1, 2], ["id", "full_name"]]
    df2 = sample_df.ww.loc[[1, 2, 3], ["signup_date", "email"]]

    combined_df = concat_columns([df1, df2])
    assert len(combined_df) == 4


def test_concat_cols_mismatched_index_adds_multiple_nans(sample_df_pandas):
    # Only pandas checks for index uniqueness
    sample_df_pandas.ww.init(index="id", logical_types={"id": "IntegerNullable"})

    df1 = sample_df_pandas.ww.loc[[0, 1], ["id", "full_name"]]
    df2 = sample_df_pandas.ww.loc[[2, 3], ["signup_date", "email"]]

    error = "Index column must be unique"
    with pytest.raises(IndexError, match=error):
        concat_columns([df1, df2])


def test_concat_cols_duplicate_columns(sample_df):
    sample_df.ww.init()
    df1 = sample_df.ww[["id", "full_name", "age"]]
    df2 = sample_df.ww[["full_name", "email", "signup_date"]]

    error = (
        "Duplicate column 'full_name' has been found in more than one input object. "
        "Please remove duplicate columns from all but one table."
    )
    with pytest.raises(ValueError, match=error):
        concat_columns([df1, df2])


def test_concat_cols_duplicate_columns_one_index(sample_df):
    sample_df.ww.init()
    df1 = sample_df.ww[["id", "full_name", "age"]]
    df2 = sample_df.ww[["id", "email", "signup_date"]]
    df2.ww.set_index("id")
    assert df1.ww.index is None
    assert df2.ww.index == "id"

    error = (
        "Duplicate column 'id' has been found in more than one input object. "
        "Please remove duplicate columns from all but one table."
    )
    with pytest.raises(ValueError, match=error):
        concat_columns([df1, df2])


def test_concat_table_order(sample_df):
    df1 = sample_df[["signup_date"]]
    df1.ww.init()
    df2 = sample_df[["id", "age"]]
    df2.ww.init(index="id")

    combined_df1 = concat_columns([df1.ww.copy(), df2.ww.copy()])
    combined_df2 = concat_columns([df2.ww.copy(), df1.ww.copy()])

    assert set(combined_df1.columns) == set(combined_df2.columns)

    assert list(combined_df1.columns) == ["signup_date", "id", "age"]
    assert list(combined_df2.columns) == ["id", "age", "signup_date"]


def test_concat_cols_all_series(sample_df):
    age_series = ww.init_series(sample_df["age"], logical_type="Double")
    combined_df_1 = concat_columns(
        [sample_df["id"], age_series, sample_df["is_registered"]],
    )
    combined_df_2 = concat_columns(
        [age_series, sample_df["id"], sample_df["is_registered"]],
    )

    assert isinstance(combined_df_1.ww.schema, ww.table_schema.TableSchema)
    assert list(combined_df_1.columns) == ["id", "age", "is_registered"]
    assert list(combined_df_2.columns) == ["age", "id", "is_registered"]
    assert isinstance(combined_df_1.ww.logical_types["age"], Double)


def test_concat_cols_row_order(sample_df):
    sample_df.ww.init(index="id")
    pd.testing.assert_index_equal(to_pandas(sample_df.index), pd.Index([0, 1, 2, 3]))

    df1 = sample_df.ww.loc[:, ["id", "full_name"]]
    df2 = sample_df.ww.loc[[2, 3, 0, 1], ["email", "phone_number"]]
    df3 = sample_df.ww.loc[
        [3, 1, 0, 2],
        [
            "age",
            "signup_date",
            "is_registered",
            "double",
            "double_with_nan",
            "integer",
            "nullable_integer",
            "boolean",
            "categorical",
            "datetime_with_NaT",
            "url",
            "ip_address",
        ],
    ]

    combined_df = concat_columns([df1, df2, df3])

    assert sample_df.ww == combined_df.ww

    # spark does not preserve index order in the same way
    if _is_spark_dataframe(sample_df):
        pd.testing.assert_index_equal(
            to_pandas(combined_df.index),
            pd.Index([0, 1, 2, 3]),
        )
    else:
        pd.testing.assert_frame_equal(to_pandas(sample_df), to_pandas(combined_df))


def test_concat_empty_list():
    error = "No objects to concatenate"
    with pytest.raises(ValueError, match=error):
        concat_columns([])


def test_concat_with_none(sample_df):
    df1 = sample_df[["id", "full_name", "email"]]
    df2 = sample_df[["phone_number", "age", "signup_date", "is_registered"]]

    error = "'NoneType' object has no attribute 'ww'"
    with pytest.raises(AttributeError, match=error):
        concat_columns([df1, df2, None])


@pytest.mark.parametrize(
    "nullable_type",
    [
        "integer",
        pytest.param(
            "boolean",
            marks=pytest.mark.xfail(reason="Bug with BooleanNullable inference."),
        ),
    ],
)
def test_concat_shorter_null_int(sample_df, nullable_type):
    """Tests that concatenating dataframes with different numbers of rows works."""
    sample_df.ww.init()
    df1 = sample_df.ww[
        [
            "email",
            "phone_number",
        ]
    ]

    # Purposefully create a dataframe with a non-nullable integer column
    # that's shorter than the one to concat with
    if _is_dask_dataframe(sample_df):
        pytest.skip(
            "Slicing dataframe with respect to rows is not supported with Dask input",
        )
    df2 = sample_df.ww[
        [
            nullable_type,
            "categorical",
        ]
    ].iloc[1:, :]
    df2.ww.init()

    result = concat_columns([df1, df2])
    expected_type = IntegerNullable if nullable_type == "integer" else BooleanNullable
    assert isinstance(result.ww.logical_types[nullable_type], expected_type)
    assert (
        result.shape[0] == df1.shape[0]
    ), "The output of concat_columns has the incorrect number of rows."
    assert (
        result.shape[1] == df1.shape[1] + df2.shape[1]
    ), "The output of concat_columns has the incorrect number of columns."
