import re
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from woodwork.accessor_utils import (
    _is_dask_dataframe,
    _is_dask_series,
    _is_spark_dataframe,
    _is_spark_series,
    init_series,
)
from woodwork.config import config
from woodwork.exceptions import (
    TypeConversionError,
    TypeConversionWarning,
    TypeValidationError,
)
from woodwork.logical_types import (
    URL,
    Age,
    AgeFractional,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    Integer,
    IntegerNullable,
    LatLong,
    Ordinal,
    PhoneNumber,
    PostalCode,
    Unknown,
    _replace_nans,
)
from woodwork.tests.testing_utils.table_utils import (
    concat_dataframe_or_series,
    to_pandas,
)
from woodwork.utils import import_or_none

ps = import_or_none("pyspark.pandas")


def test_logical_eq():
    assert Boolean == Boolean
    assert Boolean() == Boolean()
    assert Categorical != Boolean
    assert Datetime != Datetime()
    assert Datetime() == Datetime(datetime_format=None)
    assert Datetime() != Datetime(datetime_format="%Y-%m-%d")


def test_logical_repr():
    assert isinstance(repr(Boolean), str)
    assert repr(Boolean) == "Boolean"
    assert isinstance(repr(Categorical), str)
    assert repr(Categorical) == "Categorical"


def test_instantiated_type_str():
    assert str(Categorical()) == "Categorical"
    assert str(Boolean()) == "Boolean"


def test_ordinal_order_errors():
    series = pd.Series([1, 2, 3]).astype("category")

    with pytest.raises(
        TypeError,
        match="Order values must be specified in a list or tuple",
    ):
        Ordinal(order="not_valid").transform(series)

    with pytest.raises(ValueError, match="Order values cannot contain duplicates"):
        Ordinal(order=["a", "b", "b"]).transform(series)


def test_ordinal_init_with_order():
    order = ["bronze", "silver", "gold"]
    ordinal_from_list = Ordinal(order=order)
    assert ordinal_from_list.order == order
    assert str(ordinal_from_list) == "Ordinal: ['bronze', 'silver', 'gold']"

    order = ("bronze", "silver", "gold")
    ordinal_from_tuple = Ordinal(order=order)
    assert ordinal_from_tuple.order == order
    assert str(ordinal_from_list) == "Ordinal: ['bronze', 'silver', 'gold']"


def test_ordinal_transform_validates(ordinal_transform_series_pandas) -> None:
    typ = Ordinal(order=None)
    with pytest.raises(TypeError, match=r"order values defined"):
        typ.transform(ordinal_transform_series_pandas)


def test_ordinal_transform_pandas(ordinal_transform_series_pandas) -> None:
    order = [2, 1, 3]
    typ = Ordinal(order=order)
    ser_ = typ.transform(ordinal_transform_series_pandas)

    assert ser_.dtype == "category"
    pd.testing.assert_index_equal(ser_.cat.categories, pd.Index(order, dtype="int64"))


def test_ordinal_transform_dask(ordinal_transform_series_dask) -> None:
    order = [2, 1, 3]
    typ = Ordinal(order=order)
    ser_ = typ.transform(ordinal_transform_series_dask).compute()

    assert ser_.dtype == "category"
    pd.testing.assert_index_equal(ser_.cat.categories, pd.Index(order, dtype="int64"))


def test_ordinal_transform_spark(ordinal_transform_series_spark) -> None:
    order = [2, 1, 3]
    typ = Ordinal(order=order)
    ser_ = typ.transform(ordinal_transform_series_spark)

    assert ser_.dtype == pd.StringDtype()


def test_get_valid_dtype(sample_series):
    valid_dtype = Categorical._get_valid_dtype(type(sample_series))
    if _is_spark_series(sample_series):
        assert valid_dtype == "string"
    else:
        assert valid_dtype == "category"

    valid_dtype = Boolean._get_valid_dtype(type(sample_series))
    assert valid_dtype == "bool"


def test_latlong_transform(latlong_df):
    df_type = str(type(latlong_df))
    dask = "dask" in df_type
    spark = "spark" in df_type
    nan = float("nan")

    expected_data = {
        "tuple_ints": [(1.0, 2.0), (3.0, 4.0)],
        "tuple_strings": [(1.0, 2.0), (3.0, 4.0)],
        "string_tuple": [(1.0, 2.0), (3.0, 4.0)],
        "bracketless_string_tuple": [(1.0, 2.0), (3.0, 4.0)],
        "list_strings": [(1.0, 2.0), (3.0, 4.0)],
        "combo_tuple_types": [(1.0, 2.0), (3.0, 4.0)],
        "null_value": [nan, (3.0, 4.0)],
        "null_latitude": [(nan, 2.0), (3.0, 4.0)],
        "both_null": [(nan, nan), (3.0, 4.0)],
    }

    latlong = LatLong()
    for column in latlong_df:
        series = latlong_df[column]
        actual = latlong.transform(series)

        if dask:
            actual = actual.compute()
        elif spark:
            actual = actual.to_pandas()

        expected = pd.Series(expected_data[column], name=column)
        pd.testing.assert_series_equal(actual, expected)


def test_latlong_transform_empty_series(empty_latlong_df):
    latlong = LatLong()
    series = empty_latlong_df["latlong"]
    actual = latlong.transform(series)

    if _is_dask_series(actual):
        actual = actual.compute()
    elif _is_spark_series(actual):
        actual = actual.to_pandas()

    assert actual.empty
    assert actual.name == "latlong"
    assert actual.dtype == latlong.primary_dtype


def test_latlong_validate(latlong_df):
    error_message = re.escape(
        "Cannot initialize Woodwork. Series does not contain properly formatted "
        "LatLong data. Try reformatting before initializing or use the "
        "woodwork.init_series function to initialize.",
    )
    latlong = LatLong()
    series = latlong_df["tuple_ints"]
    new_series = init_series(series, logical_type=LatLong)
    latlong.validate(new_series)
    with pytest.raises(TypeValidationError, match=error_message):
        latlong.validate(series)


def test_datetime_transform(datetimes):
    datetime = Datetime()
    for series in datetimes:
        assert str(series.dtype) == "object"
        transform = datetime.transform(series)
        assert str(transform.dtype) == "datetime64[ns]"
        assert datetime.datetime_format is not None


def test_datetime_inference_ambiguous_format():
    datetime = Datetime()
    dates = pd.Series(["01/01/2017"] * 2 + ["13/12/2017"], name="dates")
    warning = (
        "Some rows in series 'dates' are incompatible with datetime format "
        "'%m/%d/%Y' and have been replaced with null values. You may be able "
        "to fix this by using an instantiated Datetime logical type with a different "
        "format string specified for this column during Woodwork initialization."
    )
    with pytest.warns(TypeConversionWarning, match=warning):
        transformed = datetime.transform(dates)
    assert str(transformed.dtype) == "datetime64[ns]"
    assert transformed[2] is pd.NaT
    assert datetime.datetime_format == "%m/%d/%Y"


def test_datetime_coerce_user_format():
    datetime = Datetime(datetime_format="%m/%d/%Y")
    dates = pd.Series(["01/01/2017"] * 2 + ["13/12/2017"], name="dates")
    warning = (
        "Some rows in series 'dates' are incompatible with datetime format "
        "'%m/%d/%Y' and have been replaced with null values. You may be able "
        "to fix this by using an instantiated Datetime logical type with a different "
        "format string specified for this column during Woodwork initialization."
    )
    with pytest.warns(TypeConversionWarning, match=warning):
        transformed = datetime.transform(dates)
    assert str(transformed.dtype) == "datetime64[ns]"
    assert transformed[2] is pd.NaT
    assert datetime.datetime_format == "%m/%d/%Y"


def test_ordinal_transform(sample_series):
    series_type = str(type(sample_series))
    dask = "dask" in series_type
    spark = "spark" in series_type

    if dask or spark:
        pytest.xfail(
            "Fails with Dask and Spark - ordinal data validation not supported",
        )

    ordinal_incomplete_order = Ordinal(order=["a", "b"])
    error_msg = re.escape(
        "Ordinal column sample_series contains values that are not "
        "present in the order values provided: ['c']",
    )

    with pytest.raises(ValueError, match=error_msg):
        ordinal_incomplete_order.transform(sample_series)


def test_ordinal_validate(sample_series):
    series_type = str(type(sample_series))
    dask = "dask" in series_type
    spark = "spark" in series_type

    if dask or spark:
        pytest.xfail(
            "Fails with Dask and Spark - ordinal data validation not supported",
        )

    ordinal_incomplete_order = Ordinal(order=["a", "b"])
    error_msg = re.escape(
        "Ordinal column sample_series contains values that are not "
        "present in the order values provided: ['c']",
    )

    with pytest.raises(ValueError, match=error_msg):
        ordinal_incomplete_order.validate(sample_series)

    new_type = "string"
    error_message = re.escape(
        f"Series dtype '{new_type}' is incompatible with ordinal LogicalType, try converting to category dtype",
    )
    with pytest.raises(TypeValidationError, match=error_message):
        ordinal_incomplete_order.validate(sample_series.astype(new_type))


def test_email_address_validate(sample_df):
    email_address = EmailAddress()
    dtype = email_address.primary_dtype
    series = sample_df["email"].astype(dtype)
    invalid_row = pd.Series({4: "bad_email"}, name="email").astype(dtype)

    if _is_spark_series(series):
        invalid_row = ps.from_pandas(invalid_row)

    assert email_address.validate(series) is None

    series = concat_dataframe_or_series(series, invalid_row).astype(dtype)

    match = "Series email contains invalid email address values. "
    match += "The email_inference_regex can be changed in the config if needed."

    with pytest.raises(TypeValidationError, match=match):
        email_address.validate(series)

    actual = email_address.validate(series, return_invalid_values=True)
    expected = pd.Series({4: "bad_email"}, name="email").astype(dtype)
    assert to_pandas(actual).equals(expected)


def test_url_validate(sample_df):
    logical_type = URL()
    dtype = logical_type.primary_dtype
    series = sample_df["url"].astype(dtype)
    invalid_row = pd.Series({4: "bad_url"}, name="url").astype(dtype)
    if _is_spark_series(series):
        invalid_row = ps.from_pandas(invalid_row)

    assert logical_type.validate(series) is None

    series = concat_dataframe_or_series(series, invalid_row).astype(dtype)

    match = "Series url contains invalid url values. "
    match += "The url_inference_regex can be changed in the config if needed."

    with pytest.raises(TypeValidationError, match=match):
        logical_type.validate(series)

    actual = logical_type.validate(series, return_invalid_values=True)
    expected = pd.Series({4: "bad_url"}, name="url").astype(dtype)
    assert to_pandas(actual).equals(expected)


@pytest.mark.parametrize(
    argnames="logical_type",
    ids=["age", "age_fractional", "age_nullable"],
    argvalues=[Age(), AgeFractional(), AgeNullable()],
)
def test_age_validate(sample_df, logical_type):
    series = sample_df["age"]
    if isinstance(logical_type, Age):
        series = series.dropna()

    dtype = logical_type.primary_dtype
    series = series.astype(dtype)

    assert logical_type.validate(series, return_invalid_values=False) is None
    invalid_row = pd.Series({4: -3}, name="age", dtype=dtype)

    if _is_spark_series(series):
        invalid_row = ps.from_pandas(invalid_row)

    series = concat_dataframe_or_series(series, invalid_row).astype(dtype)

    match = "Series age contains negative values."
    with pytest.raises(TypeValidationError, match=match):
        logical_type.validate(series, return_invalid_values=False)

    actual = logical_type.validate(series, return_invalid_values=True)
    assert to_pandas(actual).equals(to_pandas(invalid_row))


def test_phone_number_validate(sample_df):
    phone_number = PhoneNumber()
    dtype = phone_number.primary_dtype
    series = sample_df["phone_number"].astype(dtype)
    invalid_row = pd.Series({4: "bad_phone"}, name="phone_number").astype(dtype)

    if _is_spark_series(series):
        invalid_row = ps.from_pandas(invalid_row)

    assert phone_number.validate(series) is None

    series = concat_dataframe_or_series(series, invalid_row).astype(dtype)

    match = "Series phone_number contains invalid phone number values. "
    match += "The phone_inference_regex can be changed in the config if needed."

    with pytest.raises(TypeValidationError, match=match):
        phone_number.validate(series)

    actual = phone_number.validate(series, return_invalid_values=True)
    expected = pd.Series({4: "bad_phone"}, name="phone_number").astype(dtype)
    assert to_pandas(actual).equals(expected)


def test_phone_number_validate_complex(sample_df_phone_numbers):
    phone_number = PhoneNumber()
    dtype = phone_number.primary_dtype
    series = sample_df_phone_numbers["phone_number"].astype(dtype)
    # Current inference function does not match lack of area code
    invalid_row = pd.Series(
        {17: "252 9384", 18: "+1 194 129 1991", 19: "+01 236 248 8482"},
        name="phone_number",
    ).astype(dtype)

    series = concat_dataframe_or_series(series, invalid_row).astype(dtype)

    actual = phone_number.validate(series, return_invalid_values=True)
    expected = pd.Series(
        {17: "252 9384", 18: "+1 194 129 1991", 19: "+01 236 248 8482"},
        name="phone_number",
    ).astype(dtype)
    assert to_pandas(actual).equals(expected)


def test_postal_code_validate(sample_df_postal_code):
    pc = PostalCode()
    series = sample_df_postal_code["postal_code"]
    invalid_types = pd.Series(
        [
            "hello",
            "HELLO",
            "51342-HEL0",
        ],
    )

    series = concat_dataframe_or_series(series, invalid_types)

    series.name = "postal_code"
    match = "Series postal_code contains invalid postal code values. "
    match += "The postal_code_inference_regex can be changed in the config if needed."

    with pytest.raises(TypeValidationError, match=match):
        pc.validate(series)


def test_postal_code_validate_complex(sample_df_postal_code):
    pc = PostalCode()
    series = sample_df_postal_code["postal_code"]
    invalid_types = pd.Series(
        [
            "1234",
            "123455",
            "123456789",
            "1234-65433",
            "K1A0B1",  # Canadian formatting
            "K1A 0B1",
            "K1A-0B1",
            "DT1 1AE",  # UK formatting
            "DT1-1AE",
            "DT11AE",
        ],
    )
    actual = pc.validate(series, return_invalid_values=True)
    assert not len(actual)

    series = concat_dataframe_or_series(series, invalid_types)

    actual = pc.validate(series, return_invalid_values=True)
    pd.testing.assert_series_equal(actual, invalid_types)


def test_postal_code_validate_numeric(postal_code_numeric_series):
    series = init_series(postal_code_numeric_series, logical_type=PostalCode())
    actual = to_pandas(series.ww.validate_logical_type(return_invalid_values=True))
    expected = pd.Series({5: "1234567890"})

    pd.testing.assert_series_equal(
        actual,
        expected,
        check_dtype=False,
        check_categorical=False,
    )


def test_postal_code_error(postal_code_numeric_series_pandas):
    series = concat_dataframe_or_series(
        postal_code_numeric_series_pandas,
        pd.Series([1234.5]),
    )
    match = (
        "Error converting datatype for None from type float64 to type string. "
        "Please confirm the underlying data is consistent with logical type PostalCode."
    )
    with pytest.raises(TypeConversionError, match=match):
        init_series(series, logical_type=PostalCode())


def test_null_invalid_values_double():
    types = {"double": "Double"}
    invalid = "text", None, True, object
    df = pd.DataFrame({"double": [1.2, 3, "4", *invalid]})

    with pytest.raises(
        TypeConversionError,
        match="Please confirm the underlying data is consistent with logical type Double",
    ):
        df.ww.init(logical_types=types, null_invalid_values=False)

    nulls = [None] * len(invalid)
    expected = pd.DataFrame({"double": [1.2, 3.0, 4.0, *nulls]})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_values_boolean():
    types = {"data": "BooleanNullable"}
    invalid = "text", 1.2, 345, object, None
    data = [True, "False", *invalid]
    df = pd.DataFrame({"data": data})

    with pytest.raises(
        TypeConversionError,
        match="Please confirm the underlying data is consistent with logical type BooleanNullable",
    ):
        df.ww.init(logical_types=types, null_invalid_values=False)

    nulls = [None] * len(invalid)
    data = [True, False, *nulls]
    expected = pd.DataFrame({"data": pd.Series(data, dtype="boolean")})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_values_integer():
    types = {"data": "IntegerNullable"}
    invalid = "text", 6.7, object, None
    data = [1.0, "2", 345, *invalid]
    df = pd.DataFrame({"data": data})

    with pytest.raises(
        TypeConversionError,
        match="Please confirm the underlying data is consistent with logical type IntegerNullable",
    ):
        df.ww.init(logical_types=types, null_invalid_values=False)

    nulls = [None] * len(invalid)
    data = [1, 2, 345, *nulls]
    expected = pd.DataFrame({"data": pd.Series(data, dtype="Int64")})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_values_emails():
    types = {"data": "EmailAddress"}
    invalid = ["text", 6.7, object, None]
    valid = ["john.smith@example.com", "support@example.com", "team@example.com"]
    data = pd.Series(valid + invalid)
    df = pd.DataFrame({"data": data})

    df.ww.init(logical_types=types, null_invalid_values=False)
    expected = pd.DataFrame({"data": data.astype("string")})
    pd.testing.assert_frame_equal(df, expected)

    nulls = [None] * len(invalid)
    data = pd.Series(valid + nulls, dtype="string")
    expected = pd.DataFrame({"data": data})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_values_url():
    valid = [
        "https://github.com/alteryx",
        "https://twitter.com",
        "http://google.com",
    ]

    types = {"data": "URL"}
    invalid = ["text", 6.7, object, None]
    data = pd.Series(valid + invalid)
    df = pd.DataFrame({"data": data})

    df.ww.init(logical_types=types, null_invalid_values=False)
    expected = pd.DataFrame({"data": data.astype("string")})
    pd.testing.assert_frame_equal(df, expected)

    nulls = [None] * len(invalid)
    data = pd.Series(valid + nulls, dtype="string")
    expected = pd.DataFrame({"data": data})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_values_phone_number():
    valid = [
        "2002007865",
        "311-311-3156",
        "422.422.0461",
    ]

    types = {"data": "PhoneNumber"}
    invalid = ["text", 6.7, object, None]
    data = pd.Series(valid + invalid)
    df = pd.DataFrame({"data": data})

    df.ww.init(logical_types=types, null_invalid_values=False)
    expected = pd.DataFrame({"data": data.astype("string")})
    pd.testing.assert_frame_equal(df, expected)

    nulls = [None] * len(invalid)
    data = pd.Series(valid + nulls, dtype="string")
    expected = pd.DataFrame({"data": data})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_age_fractional():
    types = {"data": "AgeFractional"}
    invalid = "text", -6.7, object, None
    data = [0.34, "24.34", 45.0, *invalid]
    df = pd.DataFrame({"data": data})

    with pytest.raises(
        TypeConversionError,
        match="Please confirm the underlying data is consistent with logical type AgeFractional",
    ):
        df.ww.init(logical_types=types, null_invalid_values=False)

    nulls = [None] * len(invalid)
    data = [0.34, 24.34, 45.0, *nulls]
    expected = pd.DataFrame({"data": data})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_age_nullable():
    types = {"data": "AgeNullable"}
    invalid = "text", -6, object, None
    data = [34, "24", 45, *invalid]
    df = pd.DataFrame({"data": data})

    with pytest.raises(
        TypeConversionError,
        match="Please confirm the underlying data is consistent with logical type AgeNullable",
    ):
        df.ww.init(logical_types=types, null_invalid_values=False)

    nulls = [None] * len(invalid)
    data = pd.Series([34, 24, 45, *nulls], dtype="Int64")
    expected = pd.DataFrame({"data": data})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_latlong():
    valid = [
        (33.670914, -117.841501),
        "(40.423599, -86.921162)",
        (-45.031705, None),
        (None, None),
    ]
    types = {"data": "LatLong"}
    invalid = ["text", -6.7, object, None]
    df = pd.DataFrame({"data": valid + invalid})

    with pytest.raises(
        TypeValidationError,
        match="LatLong value is not properly formatted.",
    ):
        df.ww.init(logical_types=types, null_invalid_values=False)

    nan = float("nan")
    nulls = [nan] * len(invalid)
    data = pd.Series(
        [
            (33.670914, -117.841501),
            (40.423599, -86.921162),
            (-45.031705, nan),
            (nan, nan),
            *nulls,
        ],
    )
    expected = pd.DataFrame({"data": data})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_postal_code():
    types = {"data": "PostalCode"}
    invalid = ["text", 6.7, object, "123456"]
    valid = [90210, "60018-0123", "10010"]
    data = pd.Series(valid + invalid)
    df = pd.DataFrame({"data": data})

    df.ww.init(logical_types=types, null_invalid_values=False)
    expected = pd.DataFrame({"data": data.astype("category")})
    pd.testing.assert_frame_equal(df, expected)

    nulls = [None] * len(invalid)
    data = pd.Series(valid + nulls, dtype="string")
    expected = pd.DataFrame({"data": data.astype("category")})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


def test_null_invalid_postal_code_numeric():
    types = {"data": "PostalCode"}
    invalid = [-6.7, 60018.0123, 123456.0]
    valid = [90210, 60018.0, 10010.0]
    data = pd.Series(valid + invalid)
    df = pd.DataFrame({"data": data})

    with pytest.raises(
        TypeConversionError,
        match="Please confirm the underlying data is consistent with logical type PostalCode",
    ):
        df.ww.init(logical_types=types, null_invalid_values=False)

    nulls = [None] * len(invalid)
    data = pd.Series(valid + nulls, dtype="Int64")
    data = data.astype("string").astype("category")
    expected = pd.DataFrame({"data": data})
    df.ww.init(logical_types=types, null_invalid_values=True)
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "null_type",
    [None, pd.NA, pd.NaT, np.nan, "null", "N/A", "mix", 5],
)
@pytest.mark.parametrize("data_type", [int, float])
def test_integer_nullable(data_type, null_type):
    nullable_nums = pd.DataFrame(
        map(data_type, [1, 2, 3, 4, 5] * 20),
        columns=["num_nulls"],
    )
    nullable_nums["num_nulls"].iloc[-5:] = (
        [None, pd.NA, np.nan, "NA", "none"]
        if not isinstance(null_type, pd._libs.missing.NAType) and null_type == "mix"
        else [null_type] * 5
    )
    nullable_nums.ww.init()

    if not isinstance(null_type, int):
        assert isinstance(nullable_nums.ww.logical_types["num_nulls"], IntegerNullable)
        assert all(nullable_nums["num_nulls"][-5:].isna())
    elif data_type is int:
        assert isinstance(nullable_nums.ww.logical_types["num_nulls"], Integer)
    else:
        assert isinstance(nullable_nums.ww.logical_types["num_nulls"], Double)


@pytest.mark.parametrize(
    "null_type",
    [None, pd.NA, pd.NaT, np.nan, "null", " ", "N/A", "mix", True],
)
def test_boolean_nullable(null_type):
    nullable_bools = pd.DataFrame([True, False] * 50, columns=["bool_nulls"])
    nullable_bools["bool_nulls"].iloc[-5:] = (
        [None, pd.NA, np.nan, "NA", "none"]
        if not isinstance(null_type, pd._libs.missing.NAType) and null_type == "mix"
        else [null_type] * 5
    )
    nullable_bools.ww.init()

    if not isinstance(null_type, bool):
        assert isinstance(
            nullable_bools.ww.logical_types["bool_nulls"],
            BooleanNullable,
        )
        assert all(nullable_bools["bool_nulls"][-5:].isna())
    else:
        assert isinstance(nullable_bools.ww.logical_types["bool_nulls"], Boolean)


def test_replace_nans_same_types():
    series = pd.Series([1, 3, 5, -6, "nan"], dtype="object")

    new_series = _replace_nans(series, LatLong.primary_dtype)  # Object dtype

    assert new_series.dtype == "object"
    pd.testing.assert_series_equal(new_series, pd.Series([1, 3, 5, -6, "nan"]))

    new_series = _replace_nans(series, Double.primary_dtype)

    assert new_series.dtype == "float"
    pd.testing.assert_series_equal(new_series, pd.Series([1.0, 3.0, 5.0, -6.0, np.nan]))


def get_expected_dates(dates):
    expected = []
    for d in dates:
        if d is not None:
            dt_split = d.split(" ")
            date_groups = re.findall(r"\d+", dt_split[0])
            time_groups = None
            if len(dt_split) > 1:
                time_groups = dt_split[1]
            year = int(date_groups[0])  # gets the year
            month = int(date_groups[1])  # gets the month
            day = int(date_groups[2])  # gets the day
            if year > datetime.today().year + 10:
                year -= 100
            if year <= datetime.today().year - 90:
                year += 100
            new_dt = (
                f"{year}-{month}-{day} {time_groups}"
                if time_groups
                else f"{year}-{month}-{day}"
            )
            expected.append(new_dt)
        else:
            expected.append(d)
    return expected


@pytest.mark.parametrize("delim", ["/", "-", "."])
@pytest.mark.parametrize("dtype", ["string", "object"])
def test_datetime_pivot_point(dtype, delim):
    if dtype == "string" and delim != "/":
        pytest.skip("skipping because we don't want to overtest")
    dates = [
        "01/01/24",
        "01/01/30",
        "01/01/32",
        "01/01/36",
        "01/01/52",
        "01/01/56",
        "01/01/60",
        "01/01/72",
        "01/01/76",
        "01/01/80",
        None,
        "01/01/88",
    ]
    datetime_str = "%m/%d/%y"
    if delim != "/":
        dates = [s.replace("/", delim) if s is not None else s for s in dates]
        datetime_str = datetime_str.replace("/", delim)
    expected_values = [
        "2024-01-01",
        "2030-01-01",
        "2032-01-01",
        "2036-01-01",
        "2052-01-01",
        "2056-01-01",
        "1960-01-01",
        "1972-01-01",
        "1976-01-01",
        "1980-01-01",
        None,
        "1988-01-01",
    ]
    expected_values = get_expected_dates(expected_values)
    df = pd.DataFrame({"dates": dates}, dtype=dtype)
    df_expected = pd.DataFrame({"dates": expected_values}, dtype="datetime64[ns]")
    df.ww.init(logical_types={"dates": Datetime(datetime_format=datetime_str)})
    pd.testing.assert_frame_equal(df, df_expected)


@pytest.mark.parametrize("delim", ["/", "-", ".", ""])
def test_datetime_pivot_point_should_not_apply(delim):
    dates = [
        "01/01/1924",
        "01/01/1928",
        "01/01/1960",
        "01/01/1964",
        "01/01/1968",
        "01/01/1972",
        "01/01/2076",
        "01/01/2088",
    ]
    datetime_str = "%m/%d/%Y"
    if delim == "-":
        dates = [s.replace("/", delim) for s in dates]
        datetime_str = datetime_str.replace("/", delim)
    expected_values = [
        "1924-01-01",
        "1928-01-01",
        "1960-01-01",
        "1964-01-01",
        "1968-01-01",
        "1972-01-01",
        "2076-01-01",
        "2088-01-01",
    ]
    df = pd.DataFrame({"dates": dates})
    df_expected = pd.DataFrame({"dates": expected_values}, dtype="datetime64[ns]")
    df.ww.init(logical_types={"dates": Datetime(datetime_format=datetime_str)})
    pd.testing.assert_frame_equal(df, df_expected)


@pytest.mark.parametrize("type", ["pyspark", "dask"])
def test_pyspark_dask_series(type):
    dates = [
        "01/01/24",
        "01/01/28",
        "01/01/30",
        "01/01/32",
        "01/01/36",
        "01/01/40",
        "01/01/72",
        None,
        "01/01/88",
    ]
    datetime_str = "%m/%d/%y"
    expected_values = [
        "2024-01-01",
        "2028-01-01",
        "2030-01-01",
        "1932-01-01",
        "1936-01-01",
        "1940-01-01",
        "1972-01-01",
        None,
        "1988-01-01",
    ]
    expected_values = get_expected_dates(expected_values)
    df = pd.DataFrame({"dates": dates})
    if type == "pyspark":
        ps = pytest.importorskip(
            "pyspark.pandas",
            reason="Pyspark pandas not installed, skipping",
        )
        df = ps.from_pandas(df)
    else:
        dd = pytest.importorskip(
            "dask.dataframe",
            reason="Dask not installed, skipping",
        )
        df = dd.from_pandas(df, npartitions=2)
    df.ww.init(logical_types={"dates": Datetime(datetime_format=datetime_str)})
    df_expected = pd.DataFrame({"dates": expected_values}, dtype="datetime64[ns]")
    df = to_pandas(df)
    df.sort_index(inplace=True)
    pd.testing.assert_frame_equal(df, df_expected)


def test_datetime_pivot_point_no_format_provided():
    dates = [
        "01/01/24",
        "01/01/30",
        "01/01/32",
        "01/01/36",
        "01/01/52",
        "01/01/56",
        "01/01/60",
        "01/01/73",
        "01/01/76",
        "01/01/80",
        None,
        "01/01/88",
    ]
    expected_values = [
        "2024-01-01",
        "2030-01-01",
        "2032-01-01",
        "2036-01-01",
        "2052-01-01",
        "2056-01-01",
        "2060-01-01",
        "1973-01-01",
        "1976-01-01",
        "1980-01-01",
        None,
        "1988-01-01",
    ]
    expected_values = get_expected_dates(expected_values)
    df = pd.DataFrame({"dates": dates})
    df_expected = pd.DataFrame({"dates": expected_values}, dtype="datetime64[ns]")
    df.ww.init(logical_types={"dates": Datetime})
    pd.testing.assert_frame_equal(df, df_expected)


def test_datetime_formats_two_digit_years(datetime_different_formats):
    for format_, starting_date_ in datetime_different_formats:
        # 01/15/24, 01/15/28, 01/15/32, etc.
        dates = [starting_date_.replace("24", str(each)) for each in range(24, 90, 4)]
        format_split = format_.split(" ")
        final_format = (
            ("%Y-%m-%d " + format_split[1]) if (len(format_split) > 1) else "%Y-%m-%d"
        )
        expected_values = [
            datetime.strptime(
                starting_date_.replace("24", str(each)),
                format_,
            ).strftime(final_format)
            for each in range(24, 90, 4)
        ]
        expected_values = get_expected_dates(expected_values)
        df = pd.DataFrame({"dates": dates})
        expected_series = pd.to_datetime(expected_values).tz_localize(None)
        df_expected = pd.DataFrame({"dates": expected_series}, dtype="datetime64[ns]")
        df.ww.init(logical_types={"dates": Datetime})
        pd.testing.assert_frame_equal(df, df_expected)


def test_datetime_formats_two_digit_years_ambiguous():
    series = pd.Series([f"{i}-01-01" for i in range(10, 90)])
    series = init_series(series)
    expected_values = [f"{i}-01-01" for i in range(2010, 2090)]
    expected_values = get_expected_dates(expected_values)
    series_expected = pd.Series(expected_values, dtype="datetime64[ns]")
    pd.testing.assert_series_equal(series, series_expected)

    series = pd.Series([f"{i}.01.01" for i in range(10, 90)])
    series = init_series(series)
    expected_values = [f"{i}.01.01" for i in range(2010, 2090)]
    expected_values = get_expected_dates(expected_values)
    series_expected = pd.Series(expected_values, dtype="datetime64[ns]")
    pd.testing.assert_series_equal(series, series_expected)


@pytest.mark.parametrize("df_type", ["pandas", "dask", "spark"])
def test_boolean_other_values(df_type):
    df = pd.DataFrame(
        {
            "bool2": ["t", "f", "t", "f", "t", "t"],
            "bool3": ["T", "F", "T", "F", "F", "f"],
            "bool4": ["true", "false", "false", "false", "true", "true"],
            "bool5": ["True", "False", "False", "True", "false", "TRUE"],
            "bool8": ["YES", "NO", "YES", "yes", "no", "no"],
            "bool9": ["N", "N", "n", "y", "Y", "y"],
        },
    )
    if df_type == "spark":
        ps = pytest.importorskip(
            "pyspark.pandas",
            reason="Pyspark pandas not installed, skipping",
        )
        df = ps.from_pandas(df)
    elif df_type == "dask":
        dd = pytest.importorskip(
            "dask.dataframe",
            reason="Dask not installed, skipping",
        )
        df = dd.from_pandas(df, npartitions=1)
    df.ww.init()
    assert all([str(dtype) == "Boolean" for dtype in df.ww.logical_types.values()])


def test_boolean_cast_nulls_as():
    Boolean(cast_nulls_as=None)
    Boolean(cast_nulls_as=False)
    Boolean(cast_nulls_as=True)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Parameter `cast_nulls_as` must be either True or False, recieved {1}",
        ),
    ):
        Boolean(cast_nulls_as=1)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Parameter `cast_nulls_as` must be either True or False, recieved {'True'}",
        ),
    ):
        Boolean(cast_nulls_as="True")


@pytest.mark.parametrize("cast_null", [None, True, False])
@pytest.mark.parametrize("null", [None, pd.NA, np.nan])
@pytest.mark.parametrize(
    "series",
    (
        [True, True, False, False],
        ["Yes", "yes", "NO", "no"],
        ["True", "true", "false", "FALSE"],
    ),
)
def test_boolean_with_null_error(series, null, cast_null):
    df = pd.DataFrame({"bool_with_null": series + [null]})
    if cast_null is None:
        with pytest.raises(
            ValueError,
            match="Expected no null values in this Boolean column. If you want to keep the nulls, use BooleanNullable type. Otherwise, cast these nulls to a boolean value",
        ):
            df.ww.init(
                logical_types={"bool_with_null": Boolean(cast_nulls_as=cast_null)},
            )
    else:
        df.ww.init(logical_types={"bool_with_null": Boolean(cast_nulls_as=cast_null)})
        assert list(df["bool_with_null"].values) == [
            True,
            True,
            False,
            False,
            cast_null,
        ]


@pytest.mark.parametrize("df_type", ["pandas", "dask", "spark"])
@pytest.mark.parametrize("null", [None, pd.NA, np.nan])
def test_boolean_nullable_other_values_dont_cast(null, df_type):
    df = pd.DataFrame(
        {
            "bool1": ["N", "N", "n", null, "Y", "y"],
            "bool2": ["t", "f", "t", null, "f", "t"],
            "bool3": ["T", "F", "T", "F", "F", null],
            "bool4": ["true", "false", "false", "false", "true", null],
            "bool5": ["True", "False", "False", "True", null, "TRUE"],
            "bool7": ["YES", "NO", "YES", "yes", null, "no"],
        },
    )
    if df_type == "spark":
        ps = pytest.importorskip(
            "pyspark.pandas",
            reason="Pyspark pandas not installed, skipping",
        )
        df = ps.from_pandas(df)
    elif df_type == "dask":
        dd = pytest.importorskip(
            "dask.dataframe",
            reason="Dask not installed, skipping",
        )
        df = dd.from_pandas(df, npartitions=1)
    df.ww.init()
    assert all(
        [str(dtype) == "BooleanNullable" for dtype in df.ww.logical_types.values()],
    )


def test_boolean_mixed_string():
    df = pd.DataFrame(
        {
            "a": ["yes", "y", "n", "n"],
            "b": ["True", "t", "False", "False"],
            "c": ["0", "y", "0", "y"],
        },
    )

    df.ww.init()
    assert all([str(dtype) != "Boolean" for dtype in df.ww.logical_types.values()])


@pytest.mark.parametrize("ints_to_config,type", [[False, "Integer"], [True, "Boolean"]])
def test_boolean_int_works(ints_to_config, type):
    df = pd.DataFrame({"ints": [i % 2 for i in range(100)]})
    df2 = df.copy()

    def tester_df_ints():
        df.ww.init()
        assert [str(v) for v in df.ww.logical_types.values()] == [type]

        df2.ww.init(logical_types={"ints": "Boolean"})
        assert [str(v) for v in df2.ww.logical_types.values()] == ["Boolean"]
        assert df2.values.tolist() == [[bool(i % 2)] for i in range(100)]

    if ints_to_config:
        with config.with_options(boolean_inference_ints={0, 1}):
            tester_df_ints()
    else:
        tester_df_ints()


def test_boolean_strings_of_numeric_work():
    config_len = len(config.get_option("boolean_inference_strings"))
    expected = {
        "str_ints": pd.Series([True, False, False] * 10, dtype="bool"),
        "str_floats": pd.Series([True, False, False] * 10, dtype="bool"),
        "str_ints_nans": pd.Series([True, False, np.nan] * 10, dtype="boolean"),
        "str_floats_nans": pd.Series([True, False, np.nan] * 10, dtype="boolean"),
    }
    str_ints = ["1", "0", "0"] * 10
    str_floats = ["1.0", "0.0", "0.0"] * 10
    str_ints_nans = ["1", "0", None] * 10
    str_floats_nans = ["1.0", "0.0", None] * 10

    df = pd.DataFrame(
        {
            "str_ints": str_ints,
            "str_floats": str_floats,
            "str_ints_nans": str_ints_nans,
            "str_floats_nans": str_floats_nans,
        },
    )
    logical_types = {
        "str_ints": "Boolean",
        "str_floats": "Boolean",
        "str_ints_nans": "BooleanNullable",
        "str_floats_nans": "BooleanNullable",
    }
    df.ww.init(logical_types=logical_types)
    pd.testing.assert_frame_equal(pd.DataFrame.from_dict(expected), df)
    config_len_after = len(config.get_option("boolean_inference_strings"))
    assert config_len == config_len_after


@patch("woodwork.logical_types._coerce_boolean")
def test_coerce_boolean_not_called_for_bool_dtype(coerce_boolean_patch):
    series = pd.Series([0, 1, 1, 0, 1, 0, 1], dtype="bool")
    series_init = init_series(series)
    assert not coerce_boolean_patch.called
    assert series_init.dtype == "bool"


def test_object_dtype_inference(comprehensive_df):
    expected = {
        "ints": "Integer",
        "ints_str": "Integer",
        "ints_null": "IntegerNullable",
        "ints_null_str": "IntegerNullable",
        "floats": "Double",
        "floats_str": "Double",
        "floats_null": "Double",
        "floats_null_str": "Double",
        "int_float_mixed": "Double",
        "int_float_mixed_null": "Double",
        "bools": "Boolean",
        "bools_str": "Boolean",
        "bools_null": "BooleanNullable",
        "bools_null_str": "BooleanNullable",
        "datetimes": "Datetime",
        "datetimes_str": "Datetime",
        "datetimes_null_str": "Datetime",
    }
    df_copy = comprehensive_df.copy()
    df_copy_objects = comprehensive_df.copy()
    df_copy.ww.init()
    df_copy_objects.ww.init(
        logical_types={col: Unknown for col in df_copy_objects.columns},
    )
    if _is_dask_dataframe(df_copy):
        df_copy = df_copy.ww.compute()
        df_copy_objects = df_copy_objects.ww.compute()
    elif _is_spark_dataframe(df_copy):
        df_copy = df_copy.ww.to_pandas()
        df_copy_objects = df_copy_objects.ww.to_pandas()
    # Confirm proper Woodwork inference for pandas-inferred object columns
    assert {
        col: str(ltype) for col, ltype in df_copy.ww.logical_types.items()
    } == expected
    for col in df_copy_objects:
        df_copy_objects[col] = df_copy_objects[col].astype("object")
    df_copy_objects.ww.init()
    # Confirm proper Woodwork inference when every column is converted to string and then cast to object
    assert {
        col: str(ltype) for col, ltype in df_copy_objects.ww.logical_types.items()
    } == expected
