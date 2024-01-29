import numpy as np
import pandas as pd
import pytest

from woodwork.accessor_utils import (
    _is_dask_dataframe,
    _is_dask_series,
    _is_dataframe,
    _is_series,
    _is_spark_dataframe,
    _is_spark_series,
    get_invalid_schema_message,
    init_series,
    is_schema_valid,
)
from woodwork.exceptions import TypeConversionError
from woodwork.logical_types import Categorical, Datetime, LatLong, NaturalLanguage


def test_init_series_valid_conversion_specified_ltype(sample_series):
    if _is_spark_series(sample_series):
        sample_series = sample_series.astype("str")
    else:
        sample_series = sample_series.astype("object")

    series = init_series(sample_series, logical_type="categorical")
    assert series is not sample_series
    correct_dtype = Categorical._get_valid_dtype(type(sample_series))
    assert series.dtype == correct_dtype
    assert isinstance(series.ww.logical_type, Categorical)
    assert series.ww.semantic_tags == {"category"}

    series = init_series(sample_series, logical_type="natural_language")
    assert series is not sample_series
    correct_dtype = NaturalLanguage._get_valid_dtype(type(sample_series))
    assert series.dtype == correct_dtype
    assert isinstance(series.ww.logical_type, NaturalLanguage)
    assert series.ww.semantic_tags == set()


def test_init_series_with_pd_extension_array():
    extension_categories = pd.Categorical([1, 2, 3])
    series = init_series(extension_categories)
    pd_reference_series = init_series(pd.Series([1, 2, 3], dtype="category"))
    assert series.equals(pd_reference_series)
    assert series.ww.logical_type == pd_reference_series.ww.logical_type
    assert series.ww.semantic_tags == pd_reference_series.ww.semantic_tags

    extension_ints = pd.array(np.array([1, 2, 3, 4], dtype="int64"))
    series = init_series(extension_ints)
    pd_reference_series = init_series(pd.Series([1, 2, 3, 4], dtype="Int64"))
    assert series.equals(pd_reference_series)
    assert series.ww.logical_type == pd_reference_series.ww.logical_type
    assert series.ww.semantic_tags == pd_reference_series.ww.semantic_tags


def test_init_series_with_invalid_type(sample_df):
    inputs = [sample_df, 1, "string", None]
    for input_ in inputs:
        error_message = (
            f"Input must be of series type. The current input is of type {type(input_)}"
        )
        with pytest.raises(TypeError, match=error_message):
            init_series(input_)


def test_init_series_with_np_array(sample_series_pandas):
    series = init_series(sample_series_pandas.to_numpy())
    series2 = init_series(
        sample_series_pandas,
    )  # Sample series panda contains ['a','b','c','a']
    assert series.equals(series2)
    assert series.ww.logical_type == series2.ww.logical_type
    assert series.ww.semantic_tags == series2.ww.semantic_tags


def test_init_series_with_multidimensional_np_array():
    input_ = np.array([["a", "b"], ["a", "b"]])
    error_message = f"np.ndarray input must be 1 dimensional. Current np.ndarray is {input_.ndim} dimensional"
    with pytest.raises(ValueError, match=error_message):
        init_series(input_)


def test_init_series_valid_conversion_inferred_ltype(sample_series):
    if _is_spark_series(sample_series):
        sample_series = sample_series.astype("str")
    else:
        sample_series = sample_series.astype("object")

    series = init_series(sample_series)
    assert series is not sample_series
    correct_dtype = Categorical._get_valid_dtype(type(sample_series))
    assert series.dtype == correct_dtype
    assert isinstance(series.ww.logical_type, Categorical)
    assert series.ww.semantic_tags == {"category"}


def test_init_series_with_datetime(sample_datetime_series):
    series = init_series(sample_datetime_series, logical_type="datetime")
    assert series.dtype == "datetime64[ns]"
    assert isinstance(series.ww.logical_type, Datetime)


def test_init_series_with_latlong(latlong_df):
    for column in latlong_df:
        series = latlong_df[column]
        series = init_series(series, logical_type="LatLong")
        assert isinstance(series.ww.logical_type, LatLong)


def test_init_series_all_parameters(sample_series):
    if _is_spark_series(sample_series):
        sample_series = sample_series.astype("str")
    else:
        sample_series = sample_series.astype("object")

    metadata = {"meta_key": "meta_value"}
    description = "custom description"
    origin = "base"
    series = init_series(
        sample_series,
        logical_type="categorical",
        semantic_tags=["custom_tag"],
        metadata=metadata,
        description=description,
        origin=origin,
        use_standard_tags=False,
    )
    assert series is not sample_series
    correct_dtype = Categorical._get_valid_dtype(type(sample_series))
    assert series.dtype == correct_dtype
    assert isinstance(series.ww.logical_type, Categorical)
    assert series.ww.semantic_tags == {"custom_tag"}
    assert series.ww.metadata == metadata
    assert series.ww.description == description
    assert series.ww.origin == origin


def test_init_series_error_on_invalid_conversion(sample_series):
    if _is_dask_series(sample_series):
        pytest.xfail(
            "Dask type conversion with astype does not fail until compute is called",
        )
    if _is_spark_series(sample_series):
        pytest.xfail(
            "Spark allows this conversion, filling values it cannot convert with NaN "
            "and converting dtype to float.",
        )

    error_message = (
        "Error converting datatype for sample_series from type category to type Int64. "
        "Please confirm the underlying data is consistent with logical type IntegerNullable."
    )
    with pytest.raises(TypeConversionError, match=error_message):
        init_series(sample_series, logical_type="integer_nullable")


def test_is_series(sample_df):
    assert _is_series(sample_df["id"])
    assert not _is_series(sample_df)


def test_is_dataframe(sample_df):
    assert _is_dataframe(sample_df)
    assert not _is_dataframe(sample_df["id"])


def test_get_invalid_schema_message(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(
        name="test_schema",
        index="id",
        logical_types={"id": "Double", "full_name": "PersonFullName"},
    )
    schema = schema_df.ww.schema

    assert get_invalid_schema_message(schema_df, schema) is None
    assert (
        get_invalid_schema_message(sample_df, schema)
        == "dtype mismatch for column id between DataFrame dtype, int64, and Double dtype, float64"
    )

    sampled_df = schema_df.sample(frac=0.3)
    assert get_invalid_schema_message(sampled_df, schema) is None

    dropped_df = schema_df.drop("id", axis=1)
    assert (
        get_invalid_schema_message(dropped_df, schema)
        == "The following columns in the typing information were missing from the DataFrame: {'id'}"
    )

    renamed_df = schema_df.rename(columns={"id": "new_col"})
    assert (
        get_invalid_schema_message(renamed_df, schema)
        == "The following columns in the DataFrame were missing from the typing information: {'new_col'}"
    )


def test_get_invalid_schema_message_dtype_mismatch(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(
        logical_types={"age": "Categorical", "full_name": "PersonFullName"},
    )
    schema = schema_df.ww.schema

    incorrect_int_dtype_df = schema_df.ww.astype({"id": "Int64"})
    incorrect_bool_dtype_df = schema_df.ww.astype({"is_registered": "Int64"})

    assert (
        get_invalid_schema_message(incorrect_int_dtype_df, schema)
        == "dtype mismatch for column id between DataFrame dtype, Int64, and Integer dtype, int64"
    )
    assert (
        get_invalid_schema_message(incorrect_bool_dtype_df, schema)
        == "dtype mismatch for column is_registered between DataFrame dtype, Int64, and BooleanNullable dtype, boolean"
    )

    # Spark backup dtypes make these checks not relevant
    if not _is_spark_dataframe(sample_df):
        incorrect_str_dtype_df = schema_df.ww.astype(
            {"full_name": "object"},
        )  # wont work for spark
        incorrect_categorical_dtype_df = schema_df.ww.astype(
            {"age": "string"},
        )  # wont work for spark
        assert (
            get_invalid_schema_message(incorrect_str_dtype_df, schema)
            == "dtype mismatch for column full_name between DataFrame dtype, object, and PersonFullName dtype, string"
        )
        assert (
            get_invalid_schema_message(incorrect_categorical_dtype_df, schema)
            == "dtype mismatch for column age between DataFrame dtype, string, and Categorical dtype, category"
        )


def test_get_invalid_schema_message_index_checks(sample_df):
    if not isinstance(sample_df, pd.DataFrame):
        pytest.xfail("Index validation not performed for Dask or Spark DataFrames")

    schema_df = sample_df.copy()
    schema_df.ww.init(
        name="test_schema",
        index="id",
        logical_types={"id": "Double", "full_name": "PersonFullName"},
    )
    schema = schema_df.ww.schema

    different_underlying_index_df = schema_df.copy()
    different_underlying_index_df["id"] = pd.Series([9, 8, 7, 6], dtype="float64")
    assert (
        get_invalid_schema_message(different_underlying_index_df, schema)
        == "Index mismatch between DataFrame and typing information"
    )

    not_unique_df = schema_df.replace({3: 1})
    not_unique_df.index = not_unique_df["id"]
    not_unique_df.index.name = None
    assert (
        get_invalid_schema_message(not_unique_df, schema)
        == "Index column is not unique"
    )

    df = pd.DataFrame(
        {
            "id": pd.Series([5, 4, 3, 2], dtype="float64"),
            "col": pd.Series(["b", "b", "b", "d"], dtype="category"),
        },
    )
    df.ww.init(index="id")
    df_schema = df.ww.schema

    nan_df = df.replace({3: None})
    nan_df["id"] = nan_df["id"].astype("float64")
    nan_df = nan_df.set_index("id", drop=False)
    actual = get_invalid_schema_message(nan_df, df_schema)
    assert actual == "Index contains null values"


def test_is_schema_valid_true(sample_df):
    sample_df.ww.init(index="id", logical_types={"phone_number": "Categorical"})
    copy_df = sample_df.copy()

    assert copy_df.ww.schema is None
    assert is_schema_valid(copy_df, sample_df.ww.schema)


def test_is_schema_valid_false(sample_df):
    sample_df.ww.init()
    schema = sample_df.ww.schema

    invalid_dtype_df = sample_df.astype({"age": "float64"})
    assert not is_schema_valid(invalid_dtype_df, schema)

    missing_col_df = sample_df.drop(columns={"is_registered"})
    assert not is_schema_valid(missing_col_df, schema)


def test_is_dask_dataframe(sample_df_dask):
    assert _is_dask_dataframe(sample_df_dask)
    assert not _is_dask_dataframe(pd.DataFrame())


def test_is_dask_series(sample_series_dask):
    assert _is_dask_series(sample_series_dask)
    assert not _is_dask_series(pd.Series())


def test_is_spark_dataframe(sample_df_spark):
    assert _is_spark_dataframe(sample_df_spark)
    assert not _is_dask_dataframe(pd.DataFrame())


def test_is_spark_series(sample_series_spark):
    assert _is_spark_series(sample_series_spark)
    assert not _is_dask_series(pd.Series())
