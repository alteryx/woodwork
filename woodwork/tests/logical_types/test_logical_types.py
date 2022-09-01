import re

import numpy as np
import pandas as pd
import pytest

from woodwork.accessor_utils import _is_dask_series, _is_spark_series, init_series
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
    _replace_nans,
)
from woodwork.tests.testing_utils.table_utils import to_pandas
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
    pd.testing.assert_index_equal(ser_.cat.categories, pd.Int64Index(order))


def test_ordinal_transform_dask(ordinal_transform_series_dask) -> None:
    order = [2, 1, 3]
    typ = Ordinal(order=order)
    ser_ = typ.transform(ordinal_transform_series_dask).compute()

    assert ser_.dtype == "category"
    pd.testing.assert_index_equal(ser_.cat.categories, pd.Int64Index(order))


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
        f"Series dtype '{new_type}' is incompatible with ordinal dtype.",
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

    series = series.append(invalid_row).astype(dtype)
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

    series = series.append(invalid_row).astype(dtype)
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
    series = series.append(invalid_row).astype(dtype)

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

    series = series.append(invalid_row).astype(dtype)
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

    series = series.append(invalid_row).astype(dtype)
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
    series = series.append(invalid_types)
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
    series = series.append(invalid_types)
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
    series = postal_code_numeric_series_pandas.append(pd.Series([1234.5]))
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
    [None, pd.NA, pd.NaT, np.nan, "null", "N/A", "mix", True],
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


@pytest.mark.parametrize("delim", ["/", "-", "."])
def test_datetime_pivot_point(delim):
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
    if delim != "/":
        dates = [s.replace("/", delim) if s is not None else s for s in dates]
    expected_values = [
        "2024-01-01",
        "2030-01-01",
        "1932-01-01",
        "1936-01-01",
        "1952-01-01",
        "1956-01-01",
        "1960-01-01",
        "1972-01-01",
        "1976-01-01",
        "1980-01-01",
        None,
        "1988-01-01",
    ]
    df = pd.DataFrame({"dates": dates})
    df_expected = pd.DataFrame({"dates": expected_values}, dtype="datetime64[ns]")
    df.ww.init(logical_types={"dates": Datetime})
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
    if delim == "-":
        dates = [s.replace("/", delim) for s in dates]
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
    df.ww.init(logical_types={"dates": Datetime})
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
    df.ww.init(logical_types={"dates": Datetime})
    df_expected = pd.DataFrame({"dates": expected_values}, dtype="datetime64[ns]")
    df = to_pandas(df)
    assert df.values.tolist() == df_expected.values.tolist()
