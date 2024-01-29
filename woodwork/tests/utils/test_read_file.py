import os
from unittest.mock import patch

import pandas as pd
import pytest

import woodwork as ww
from woodwork.logical_types import _replace_nans
from woodwork.serializers.orc_serializer import save_orc_file


def test_read_file_errors_no_content_type(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, "sample")
    sample_df_pandas.to_csv(filepath, index=False)

    no_type_error = (
        "Content type could not be inferred. Please specify content_type and try again."
    )
    with pytest.raises(RuntimeError, match=no_type_error):
        ww.read_file(filepath=filepath)


def test_read_file_errors_unsupported(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, "sample")
    sample_df_pandas.to_feather(filepath)

    content_type = "notacontenttype"
    not_supported_error = (
        "Reading from content type {} is not currently supported".format(content_type)
    )
    with pytest.raises(RuntimeError, match=not_supported_error):
        ww.read_file(filepath=filepath, content_type=content_type)


@patch("woodwork.table_accessor._validate_accessor_params")
def test_read_file_validation_control(
    mock_validate_accessor_params,
    sample_df_pandas,
    tmpdir,
):
    filepath = os.path.join(tmpdir, "sample.csv")
    sample_df_pandas.to_csv(filepath, index=False)

    assert not mock_validate_accessor_params.called
    ww.read_file(filepath=filepath, validate=False)
    assert not mock_validate_accessor_params.called
    ww.read_file(filepath=filepath)
    assert mock_validate_accessor_params.called


@pytest.mark.parametrize(
    "filepath, exportfn, kwargs, pandas_nullable_fix",
    [
        ("sample.csv", ("to_csv", {"index": False}), {}, True),
        ("sample.csv", ("to_csv", {"index": False}), {"content_type": "csv"}, True),
        (
            "sample.csv",
            ("to_csv", {"index": False}),
            {
                "index": "id",
                "time_index": "signup_date",
                "logical_types": {
                    "full_name": "NaturalLanguage",
                    "phone_number": "PhoneNumber",
                    "is_registered": "BooleanNullable",
                    "age": "IntegerNullable",
                    "nullable_integer": "IntegerNullable",
                },
                "semantic_tags": {
                    "age": ["tag1", "tag2"],
                    "is_registered": ["tag3", "tag4"],
                },
                "column_origins": {
                    "full_name": "csv",
                    "phone_number": "base",
                    "is_registered": "engineered",
                },
            },
            False,
        ),
        (
            "sample.csv",
            ("to_csv", {"index": False}),
            {
                "nrows": 3,
                "dtype": {
                    "age": "Int64",
                    "is_registered": "boolean",
                    "nullable_integer": "Int64",
                },
            },
            False,
        ),
        ("sample.feather", ("to_feather", {}), {}, False),
        (
            "sample.feather",
            ("to_feather", {}),
            {"content_type": "feather", "index": "id"},
            False,
        ),
        (
            "sample.feather",
            ("to_feather", {}),
            {"content_type": "application/feather", "index": "id"},
            False,
        ),
        ("sample.arrow", ("to_feather", {}), {}, False),
        (
            "sample.arrow",
            ("to_feather", {}),
            {"content_type": "arrow", "index": "id"},
            False,
        ),
        (
            "sample.arrow",
            ("to_feather", {}),
            {"content_type": "application/arrow", "index": "id"},
            False,
        ),
        ("sample.parquet", ("to_parquet", {}), {}, False),
        (
            "sample.parquet",
            ("to_parquet", {}),
            {"content_type": "parquet", "index": "id", "use_nullable_dtypes": True},
            False,
        ),
        (
            "sample.parquet",
            ("to_parquet", {}),
            {
                "content_type": "application/parquet",
                "index": "id",
                "use_nullable_dtypes": True,
            },
            False,
        ),
        ("sample.orc", (save_orc_file, {}), {}, True),
        (
            "sample.orc",
            (save_orc_file, {}),
            {"content_type": "orc", "index": "id"},
            True,
        ),
        (
            "sample.orc",
            (save_orc_file, {}),
            {"content_type": "application/orc", "index": "id"},
            True,
        ),
    ],
)
def test_read_file(
    sample_df_pandas,
    tmpdir,
    filepath,
    exportfn,
    kwargs,
    pandas_nullable_fix,
):
    filepath = os.path.join(tmpdir, filepath)
    func, func_kwargs = exportfn
    if isinstance(func, str):
        getattr(sample_df_pandas, func)(filepath, **func_kwargs)
    else:
        # Call save_orc_file to save orc data since pandas does not have a to_orc method
        func(sample_df_pandas, filepath, **func_kwargs)
    df = ww.read_file(filepath=filepath, **kwargs)
    assert isinstance(df.ww.schema, ww.table_schema.TableSchema)

    schema_df = sample_df_pandas.copy()
    if pandas_nullable_fix:
        # pandas does not read data into nullable types currently from csv or orc,
        # so the types in df will be different than the types inferred from sample_df_pandas
        # which uses the nullable types
        schema_df = schema_df.astype(
            {
                "age": "float64",
                "nullable_integer": "float64",
                "is_registered": "object",
            },
        )

    if func in [
        "to_csv",
        save_orc_file,
    ]:  # categorical column not inferred as categorical
        schema_df["categorical"] = schema_df["categorical"].astype("string")

    schema_df.ww.init(
        index=kwargs.get("index"),
        time_index=kwargs.get("time_index"),
        logical_types=kwargs.get("logical_types"),
        semantic_tags=kwargs.get("semantic_tags"),
        column_origins=kwargs.get("column_origins"),
    )
    if func == "to_csv":
        for c in ["signup_date", "datetime_with_NaT"]:
            df.ww.logical_types[
                c
            ].datetime_format = None  # read_csv reads datetimes as strings and infers datetime during transform

    assert df.ww.schema == schema_df.ww.schema

    if "nrows" in kwargs:
        assert len(df) == kwargs["nrows"]
        schema_df = schema_df.head(kwargs["nrows"])

    pd.testing.assert_frame_equal(df, schema_df)


def test_replace_nan_strings():
    data = {
        "double": ["<NA>", "6.2", "4.2", "3.11"],
        "integer": ["<NA>", "6", "4", "3"],
        "null": ["<NA>", "", "nan", None],
        "null_string": pd.Series(["<NA>", "", "nan", ""], dtype="string"),
        "Int64": pd.Series([1, 2, 3, 4], dtype="Int64"),
        "Float64": pd.Series([1.1, 2.2, 3.3, 4.4], dtype="Float64"),
        "boolean": pd.Series([True, True, False, False], dtype="boolean"),
        "int64": pd.Series([1, 2, 3, 4], dtype="int64"),
        "double2": pd.Series([1, 2, 3, 4.5], dtype="float64"),
        "bool": pd.Series([True, True, False, False], dtype="bool"),
        "category": pd.Series([1, 2, 2, 1], dtype="category"),
    }

    expected_null_count = {
        "double": 1,
        "integer": 1,
        "null": 4,
        "null_string": 4,
        "Int64": 0,
        "Float64": 0,
        "boolean": 0,
        "int64": 0,
        "double2": 0,
        "bool": 0,
        "category": 0,
    }

    df = pd.DataFrame(data=data)
    replaced_df = df.apply(_replace_nans)
    for col in replaced_df:
        assert replaced_df[col].isnull().sum() == expected_null_count[col]


def test_replace_nan_strings_with_read_file(tmpdir):
    filepath = os.path.join(tmpdir, "data.parquet")
    content_type = "application/parquet"

    data = {
        "double": ["<NA>", "6.2", "4.2", "3.11"],
        "integer": ["<NA>", "6", "4", "3"],
        "null": ["<NA>", "", None, "nan"],
    }

    df = pd.DataFrame(data=data)
    df.to_parquet(filepath)

    # Without replacement
    actual = ww.read_file(
        content_type=content_type,
        filepath=filepath,
        replace_nan=False,
    )

    assert actual.isnull().sum().sum() == 3

    # With replacement
    actual = ww.read_file(
        content_type=content_type,
        filepath=filepath,
        replace_nan=True,
    )
    assert actual.isnull().sum().sum() == 6
