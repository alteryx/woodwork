import re

import pandas as pd
import pytest

from woodwork.accessor_utils import _is_spark_series, init_series
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
    Categorical,
    Datetime,
    EmailAddress,
    LatLong,
    Ordinal,
    PhoneNumber,
    PostalCode,
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
        TypeError, match="Order values must be specified in a list or tuple"
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


def test_latlong_validate(latlong_df):
    error_message = re.escape(
        "Cannot initialize Woodwork. Series does not contain properly formatted "
        "LatLong data. Try reformatting before initializing or use the "
        "woodwork.init_series function to initialize."
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
            "Fails with Dask and Spark - ordinal data validation not supported"
        )

    ordinal_incomplete_order = Ordinal(order=["a", "b"])
    error_msg = re.escape(
        "Ordinal column sample_series contains values that are not "
        "present in the order values provided: ['c']"
    )

    with pytest.raises(ValueError, match=error_msg):
        ordinal_incomplete_order.transform(sample_series)


def test_ordinal_validate(sample_series):
    series_type = str(type(sample_series))
    dask = "dask" in series_type
    spark = "spark" in series_type

    if dask or spark:
        pytest.xfail(
            "Fails with Dask and Spark - ordinal data validation not supported"
        )

    ordinal_incomplete_order = Ordinal(order=["a", "b"])
    error_msg = re.escape(
        "Ordinal column sample_series contains values that are not "
        "present in the order values provided: ['c']"
    )

    with pytest.raises(ValueError, match=error_msg):
        ordinal_incomplete_order.validate(sample_series)

    new_type = "string"
    error_message = re.escape(
        f"Series dtype '{new_type}' is incompatible with ordinal dtype."
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
        ]
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
        ]
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
