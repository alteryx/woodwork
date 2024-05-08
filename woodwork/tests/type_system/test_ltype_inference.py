from unittest.mock import patch

import woodwork as ww
from woodwork.logical_types import (
    URL,
    Boolean,
    BooleanNullable,
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    Integer,
    IntegerNullable,
    IPAddress,
    LogicalType,
    NaturalLanguage,
    PhoneNumber,
    PostalCode,
    Timedelta,
    Unknown,
)
from woodwork.type_sys.type_system import (
    DEFAULT_INFERENCE_FUNCTIONS,
    DEFAULT_RELATIONSHIPS,
    DEFAULT_TYPE,
    TypeSystem,
)


def test_integer_inference(integers):
    dtypes = ["int8", "int16", "int32", "int64", "intp", "int", "Int64"]

    for series in integers:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, Integer)


def test_double_inference(doubles):
    dtypes = ["float", "float32", "float64", "float_"]

    for series in doubles:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, Double)


def test_boolean_inference(bools):
    dtypes = ["bool", "boolean"]
    for series in bools:
        for dtype in dtypes:
            cast_series = series
            if True in series.dropna().values:
                cast_series = series.astype(dtype)
            inferred_type = ww.type_system.infer_logical_type(cast_series)
            if cast_series.isnull().any():
                assert isinstance(inferred_type, BooleanNullable)
            else:
                assert isinstance(inferred_type, Boolean)


def test_datetime_inference(datetimes):
    dtypes = ["object", "string", "datetime64[ns]"]

    for series in datetimes:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, Datetime)


def test_email_inference(emails):
    dtypes = ["object", "string"]

    for series in emails:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, EmailAddress)


def test_email_inference_failure(bad_emails):
    dtypes = ["object", "string"]

    for series in bad_emails:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert not isinstance(inferred_type, EmailAddress)


def test_categorical_inference(categories):
    dtypes = ["object", "string", "category"]
    for ind, series in enumerate(categories):
        if ind == len(categories) - 1:
            dtypes = ["string", "category"]
        for dtype in dtypes:
            expected_ltype = Categorical
            if ind in [1, 3] and dtype == "object":
                expected_ltype = Integer
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, expected_ltype)


def test_postal_inference(postal):
    dtypes = ["category", "string"]
    for series in postal:
        for dtype in dtypes:
            inferred_dtype = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_dtype, PostalCode)


def test_natural_language_inference(natural_language):
    dtypes = ["object", "string"]
    for series in natural_language:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, NaturalLanguage)


@patch("woodwork.type_sys.inference_functions.natural_language_func")
def test_nl_inference_called_on_no_other_matches(nl_mock, natural_language):
    assert isinstance(
        ww.type_system.infer_logical_type(natural_language[0]),
        NaturalLanguage,
    )
    new_type_sys = TypeSystem(
        inference_functions=DEFAULT_INFERENCE_FUNCTIONS,
        relationships=DEFAULT_RELATIONSHIPS,
        default_type=DEFAULT_TYPE,
    )
    new_type_sys.inference_functions[NaturalLanguage] = nl_mock
    _ = new_type_sys.infer_logical_type(natural_language[0])
    assert nl_mock.called


@patch("woodwork.type_sys.inference_functions.natural_language_func")
def test_nl_inference_called_with_unknown_type(nl_mock, strings):
    assert isinstance(ww.type_system.infer_logical_type(strings[0]), Unknown)
    new_type_sys = TypeSystem(
        inference_functions=DEFAULT_INFERENCE_FUNCTIONS,
        relationships=DEFAULT_RELATIONSHIPS,
        default_type=DEFAULT_TYPE,
    )
    new_type_sys.inference_functions[NaturalLanguage] = nl_mock
    _ = new_type_sys.infer_logical_type(strings[0])
    assert nl_mock.called


@patch("woodwork.type_sys.inference_functions.natural_language_func")
def test_nl_inference_not_called_with_other_matches(nl_mock, integers):
    assert isinstance(ww.type_system.infer_logical_type(integers[0]), Integer)
    new_type_sys = TypeSystem(
        inference_functions=DEFAULT_INFERENCE_FUNCTIONS,
        relationships=DEFAULT_RELATIONSHIPS,
        default_type=DEFAULT_TYPE,
    )
    new_type_sys.inference_functions[NaturalLanguage] = nl_mock
    _ = new_type_sys.infer_logical_type(integers[0])
    assert not nl_mock.called


def test_categorical_inference_based_on_dtype(categories_dtype):
    """
    This test specifically targets the case in which a series can be inferred
    to be categorical strictly from its pandas dtype, but would otherwise be
    inferred as some other type.
    """
    inferred_type = ww.type_system.infer_logical_type(categories_dtype["cat"])
    assert isinstance(inferred_type, Categorical)

    inferred_type = ww.type_system.infer_logical_type(categories_dtype["non_cat"])
    assert not isinstance(inferred_type, Categorical)


def test_categorical_integers_inference(integers):
    with ww.config.with_options(numeric_categorical_threshold=0.5):
        dtypes = ["int8", "int16", "int32", "int64", "intp", "int", "Int64"]
        for series in integers:
            for dtype in dtypes:
                inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
                assert isinstance(inferred_type, Categorical)


def test_categorical_double_inference(doubles):
    with ww.config.with_options(numeric_categorical_threshold=0.5):
        dtypes = ["float", "float32", "float64", "float_"]
        for series in doubles:
            for dtype in dtypes:
                inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
                assert isinstance(inferred_type, Categorical)


def test_timedelta_inference(timedeltas):
    dtypes = ["timedelta64[ns]"]
    for series in timedeltas:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, Timedelta)


def test_unknown_inference(strings):
    dtypes = ["object", "string"]

    for series in strings:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, Unknown)


def test_unknown_inference_all_null(nulls):
    dtypes = ["object", "string", "category", "datetime64[ns]"]

    for ind, series in enumerate(nulls):
        if ind == len(nulls) - 1:
            dtypes = ["object", "string", "category"]
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            inferred_type.transform(series)
            assert isinstance(inferred_type, Unknown)


def test_unknown_inference_empty_series(empty_series):
    inferred_type = ww.type_system.infer_logical_type(empty_series.astype("string"))
    assert isinstance(inferred_type, Unknown)


def test_pdna_inference(pdnas):
    expected_logical_types = [
        NaturalLanguage,
        Unknown,
        IntegerNullable,
        BooleanNullable,
    ]

    for index, series in enumerate(pdnas):
        inferred_type = ww.type_system.infer_logical_type(series)
        assert isinstance(inferred_type, expected_logical_types[index])


def test_updated_ltype_inference(integers, type_sys):
    inference_fn = type_sys.inference_functions[ww.logical_types.Integer]
    type_sys.remove_type(ww.logical_types.Integer)

    class Integer(LogicalType):
        primary_dtype = "string"

    type_sys.add_type(Integer, inference_function=inference_fn)

    dtypes = ["int8", "int16", "int32", "int64", "intp", "int", "Int64"]

    for series in integers:
        for dtype in dtypes:
            inferred_type = type_sys.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, Integer)
            assert inferred_type.primary_dtype == "string"


def test_inference_randomly_sampled(large_df, type_sys):
    large_df.ww.init()

    inferred_type = large_df.ww.logical_types["int_nullable"]
    assert isinstance(inferred_type, IntegerNullable)
    inferred_type = large_df.ww.logical_types["bool_nullable"]
    assert isinstance(inferred_type, BooleanNullable)
    inferred_type = large_df.ww.logical_types["floats"]
    assert isinstance(inferred_type, Double)
    inferred_type = large_df.ww.logical_types["constant"]
    assert isinstance(inferred_type, Unknown)


def test_url_inference(urls):
    dtypes = ["object", "string"]

    for series in urls:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, URL)


def test_phone_inference(phone):
    dtypes = ["object", "string"]

    for series in phone:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, PhoneNumber)


def test_ip_inference(ip):
    dtypes = ["object", "string"]

    for series in ip:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert isinstance(inferred_type, IPAddress)
