import re

import pandas as pd
import pytest

from woodwork.accessor_utils import _is_koalas_series
from woodwork.exceptions import TypeConversionWarning
from woodwork.logical_types import Boolean, Categorical, Datetime, LatLong, Ordinal


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

    order = ("bronze", "silver", "gold")
    ordinal_from_tuple = Ordinal(order=order)
    assert ordinal_from_tuple.order == order


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


def test_ordinal_transform_koalas(ordinal_transform_series_koalas) -> None:
    order = [2, 1, 3]
    typ = Ordinal(order=order)
    ser_ = typ.transform(ordinal_transform_series_koalas)

    assert ser_.dtype == pd.StringDtype()


def test_get_valid_dtype(sample_series):
    valid_dtype = Categorical._get_valid_dtype(type(sample_series))
    if _is_koalas_series(sample_series):
        assert valid_dtype == "string"
    else:
        assert valid_dtype == "category"

    valid_dtype = Boolean._get_valid_dtype(type(sample_series))
    assert valid_dtype == "bool"


def test_latlong_transform(latlong_df):
    df_type = str(type(latlong_df))
    dask = "dask" in df_type
    koalas = "koalas" in df_type
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
        elif koalas:
            actual = actual.to_pandas()

        actual = actual.apply(pd.Series)
        series = pd.Series(expected_data[column])
        expected = series.apply(pd.Series)
        assert actual.equals(expected)


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
    koalas = "koalas" in series_type

    if dask or koalas:
        pytest.xfail(
            "Fails with Dask and Koalas - ordinal data validation not supported"
        )

    ordinal_incomplete_order = Ordinal(order=["a", "b"])
    error_msg = re.escape(
        "Ordinal column sample_series contains values that are not "
        "present in the order values provided: ['c']"
    )

    with pytest.raises(ValueError, match=error_msg):
        ordinal_incomplete_order.transform(sample_series)
