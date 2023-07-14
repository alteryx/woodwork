from timeit import default_timer as timer
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from packaging import version

import woodwork as ww
from woodwork.exceptions import TypeValidationError
from woodwork.logical_types import (
    Age,
    AgeFractional,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Categorical,
    CountryCode,
    CurrencyCode,
    Datetime,
    Double,
    Integer,
    IntegerNullable,
    Ordinal,
    PostalCode,
    SubRegionCode,
)
from woodwork.tests.testing_utils import concat_dataframe_or_series
from woodwork.type_sys.type_system import DEFAULT_INFERENCE_FUNCTIONS
from woodwork.type_sys.utils import (
    _get_specified_ltype_params,
    _is_categorical_series,
    _is_numeric_series,
    col_is_datetime,
    list_logical_types,
    list_semantic_tags,
)
from woodwork.utils import (
    CallbackCaller,
    _coerce_to_float,
    _convert_input_to_set,
    _get_column_logical_type,
    _infer_datetime_format,
    _is_latlong_nan,
    _is_nan,
    _is_s3,
    _is_url,
    _is_valid_latlong_series,
    _is_valid_latlong_value,
    _parse_logical_type,
    _reformat_to_latlong,
    camel_to_snake,
    get_valid_mi_types,
    import_or_none,
    import_or_raise,
)

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


def test_camel_to_snake():
    test_items = {
        "PostalCode": "postal_code",
        "SubRegionCode": "sub_region_code",
        "NaturalLanguage": "natural_language",
        "Categorical": "categorical",
    }

    for key, value in test_items.items():
        assert camel_to_snake(key) == value


def test_convert_input_to_set():
    error_message = "semantic_tags must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _convert_input_to_set(int)

    error_message = "test_text must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _convert_input_to_set({"index": {}, "time_index": {}}, "test_text")

    error_message = "include parameter must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        _convert_input_to_set(["index", 1], "include parameter")

    semantic_tags_from_single = _convert_input_to_set("index", "include parameter")
    assert semantic_tags_from_single == {"index"}

    semantic_tags_from_list = _convert_input_to_set(["index", "numeric", "category"])
    assert semantic_tags_from_list == {"index", "numeric", "category"}

    semantic_tags_from_set = _convert_input_to_set(
        {"index", "numeric", "category"},
        "include parameter",
    )
    assert semantic_tags_from_set == {"index", "numeric", "category"}


@patch("woodwork.utils._validate_string_tags")
@patch("woodwork.utils._validate_tags_input_type")
def test_validation_methods_called(mock_validate_input_type, mock_validate_strings):
    assert not mock_validate_input_type.called
    assert not mock_validate_strings.called

    _convert_input_to_set("test_tag", validate=False)
    assert not mock_validate_input_type.called

    _convert_input_to_set("test_tag", validate=True)
    assert mock_validate_input_type.called

    _convert_input_to_set(["test_tag", "tag2"], validate=False)
    assert not mock_validate_strings.called

    _convert_input_to_set(["test_tag", "tag2"], validate=True)
    assert mock_validate_strings.called


def test_list_logical_types_default():
    all_ltypes = ww.logical_types.LogicalType.__subclasses__()

    df = list_logical_types()

    assert set(df.columns) == {
        "name",
        "type_string",
        "description",
        "physical_type",
        "standard_tags",
        "is_default_type",
        "is_registered",
        "parent_type",
    }

    assert len(all_ltypes) == len(df)
    default_types_set = {str(cls) for cls in DEFAULT_INFERENCE_FUNCTIONS.keys()}
    listed_as_default = set(df[df["is_default_type"]]["name"])
    assert listed_as_default == default_types_set


def test_list_logical_types_customized_type_system():
    ww.type_system.remove_type("URL")

    class CustomRegistered(ww.logical_types.LogicalType):
        primary_dtype = "int64"

    class CustomNotRegistered(ww.logical_types.LogicalType):
        primary_dtype = "int64"

    ww.type_system.add_type(CustomRegistered)
    all_ltypes = ww.logical_types.LogicalType.__subclasses__()
    df = list_logical_types()
    assert len(all_ltypes) == len(df)
    # Check that URL is unregistered
    url = df[df.name == "URL"].iloc[0]
    assert url.is_default_type
    assert not url.is_registered

    # Check that new registered type is present and shows as registered
    index = df.name == "CustomRegistered"
    assert index.any()
    custom = df[index].iloc[0]
    assert not custom.is_default_type
    assert custom.is_registered

    # Check that new unregistered type is present and shows as not registered
    index = df.name == "CustomNotRegistered"
    assert index.any()
    custom = df[index].iloc[0]
    assert not custom.is_default_type
    assert not custom.is_registered
    ww.type_system.reset_defaults()


def test_list_semantic_tags():
    df = list_semantic_tags()

    assert set(df.columns) == {"name", "is_standard_tag", "valid_logical_types"}

    for name, valid_ltypes in df[["name", "valid_logical_types"]].values:
        if name in ["passthrough", "ignore", "index"]:
            assert valid_ltypes == "Any LogicalType"
        elif name not in ["time_index", "date_of_birth"]:
            assert isinstance(valid_ltypes, list)
            for log_type in valid_ltypes:
                assert name in log_type.standard_tags


def test_is_numeric_datetime_series(time_index_df):
    assert _is_numeric_series(time_index_df["ints"], None)
    assert _is_numeric_series(time_index_df["ints"], Double)
    assert not _is_numeric_series(time_index_df["ints"], Categorical)
    assert _is_numeric_series(time_index_df["ints"], Datetime)

    assert _is_numeric_series(time_index_df["strs"], Integer)
    assert not _is_numeric_series(time_index_df["strs"], "Categorical")
    assert not _is_numeric_series(time_index_df["strs"], Categorical)
    assert _is_numeric_series(time_index_df["strs"], Double)
    assert _is_numeric_series(time_index_df["strs"], "Double")

    assert not _is_numeric_series(time_index_df["bools"], None)
    assert not _is_numeric_series(time_index_df["bools"], "Boolean")

    assert not _is_numeric_series(time_index_df["times"], None)
    assert not _is_numeric_series(time_index_df["times"], Datetime)

    assert not _is_numeric_series(time_index_df["letters"], None)
    assert not _is_numeric_series(time_index_df["letters"], Double)
    assert not _is_numeric_series(time_index_df["letters"], Categorical)


def test_get_ltype_params():
    params_empty_class = _get_specified_ltype_params(Categorical)
    assert params_empty_class == {}
    params_empty = _get_specified_ltype_params(Categorical())
    assert params_empty == {}

    params_class = _get_specified_ltype_params(Datetime)
    assert params_class == {}

    params_null = _get_specified_ltype_params(Datetime())
    assert params_null == {"datetime_format": None, "timezone": None}

    ymd = "%Y-%m-%d"
    params_value = _get_specified_ltype_params(
        Datetime(datetime_format=ymd, timezone="UTC"),
    )
    assert params_value == {"datetime_format": ymd, "timezone": "UTC"}


def test_import_or_raise():
    assert import_or_raise("pandas", "Module pandas could not be found") == pd

    error = "Module nonexistent could not be found."
    with pytest.raises(ImportError, match=error):
        import_or_raise("nonexistent", error)


def test_import_or_none():
    assert import_or_none("pandas") == pd
    assert import_or_none("nonexistent") is None


def test_is_url():
    assert _is_url("https://www.google.com/")
    assert not _is_url("google.com")


def test_is_s3():
    assert _is_s3("s3://test-bucket/test-key")
    assert not _is_s3("https://woodwork-static.s3.amazonaws.com/")


@pytest.mark.parametrize(
    "test_input, error_msg",
    [
        ({1, 2, 3}, "LatLong value is not properly formatted."),
        ("{1, 2, 3}", "LatLong value is not properly formatted."),
        ("This is text", "LatLong value is not properly formatted."),
        ("'(1,2)'", "LatLong value is not properly formatted."),
        ((1, 2, 3), "LatLong values must have exactly two values"),
        ("(1, 2, 3)", "LatLong values must have exactly two values"),
        ("(1,)", "LatLong values must have exactly two values"),
        (
            ("41deg52'54\" N", "21deg22'54\" W"),
            "LatLong values must be in decimal degrees.",
        ),
        ((41.5, "21deg22'54\" W"), "LatLong values must be in decimal degrees."),
    ],
)
def test_reformat_to_latlong_errors(test_input, error_msg):
    with pytest.raises(TypeValidationError, match=error_msg):
        _reformat_to_latlong(test_input)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((1, 2), (1, 2)),
        (("1", "2"), (1, 2)),
        ("(1,2)", (1, 2)),
        ([1, 2], (1, 2)),
        (["1", "2"], (1, 2)),
        ("[1, 2]", (1, 2)),
        ("1, 2", (1, 2)),
        ((1, np.nan), (1, np.nan)),
        ((np.nan, "1"), (np.nan, 1)),
        ("(1, nan)", (1, np.nan)),
        ("(NaN, 9)", (np.nan, 9)),
        ("(1, None)", (1, np.nan)),
        ("(<NA>, 9)", (np.nan, 9)),
        ((np.nan, np.nan), (np.nan, np.nan)),
        ((pd.NA, pd.NA), (np.nan, np.nan)),
        ((None, None), (np.nan, np.nan)),
        (None, np.nan),
        (np.nan, np.nan),
        (pd.NA, np.nan),
        ("None", np.nan),
        ("NaN", np.nan),
        ("<NA>", np.nan),
    ],
)
@pytest.mark.parametrize("is_spark", [True, False])
def test_reformat_to_latlong(test_input, expected, is_spark):
    if isinstance(expected, (list, tuple)):
        if is_spark:
            assert _reformat_to_latlong(test_input, is_spark) == list(expected)
        else:
            assert _reformat_to_latlong(test_input, is_spark) == expected
    else:
        assert _reformat_to_latlong(test_input) is expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (4, 4.0),
        ("2.2", 2.2),
        (None, np.nan),
        (np.nan, np.nan),
        (pd.NA, np.nan),
    ],
)
def test_coerce_to_float(test_input, expected):
    if np.isnan(expected):
        assert _coerce_to_float(test_input) is expected
    else:
        assert _coerce_to_float(test_input) == expected


def test_is_nan():
    assert _is_nan(None)
    assert _is_nan(np.nan)
    assert _is_nan(pd.NA)
    assert _is_nan("None")
    assert _is_nan("nan")
    assert _is_nan("NaN")
    assert _is_nan("<NA>")
    assert _is_nan("")
    assert _is_nan("null")

    assert not _is_nan([None, 1, 3])
    assert not _is_nan([])
    assert not _is_nan(0)
    assert not _is_nan(False)
    assert not _is_nan({"key": "value"})


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((1.0, 2.0), True),
        ((1.0, np.nan), True),
        ((np.nan, 2.0), True),
        ((np.nan, np.nan), True),
        (np.nan, True),
        (pd.NA, False),
        (2.0, False),
        ([2.0], False),
        ([None, None], False),
        ("None", False),
        ([1.0, 2.0], False),
        ((pd.NA, pd.NA), True),
        (("a", 2.0), False),
        ((1.0, 2.0, 3.0), False),
        (None, False),
    ],
)
def test_is_valid_latlong_value(test_input, expected):
    assert _is_valid_latlong_value(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ([1.0, 2.0], True),
        ([1.0, np.nan], True),
        ([np.nan, 2.0], True),
        ([np.nan, np.nan], True),
        (np.nan, True),
        (None, True),
        (pd.NA, False),
        (2.0, False),
        ([2.0], False),
        ([None, None], True),
        ("None", False),
        ((1.0, 2.0), False),
        ((pd.NA, pd.NA), False),
        (("a", 2.0), False),
        ((1.0, 2.0, 3.0), False),
    ],
)
def test_is_valid_latlong_value_spark(test_input, expected):
    assert _is_valid_latlong_value(test_input, is_spark=True) == expected


def test_is_valid_latlong_series():
    valid_series = pd.Series([(1.0, 2.0), (3.0, 4.0)])
    invalid_series = pd.Series([(1.0, 2.0), (3.0, "4.0")])

    assert _is_valid_latlong_series(valid_series) is True
    assert _is_valid_latlong_series(invalid_series) is False


def test_get_valid_mi_types():
    valid_types = get_valid_mi_types()
    expected_types = [
        Age,
        AgeFractional,
        AgeNullable,
        Boolean,
        BooleanNullable,
        Categorical,
        CountryCode,
        CurrencyCode,
        Datetime,
        Double,
        Integer,
        IntegerNullable,
        Ordinal,
        PostalCode,
        SubRegionCode,
    ]

    assert valid_types == expected_types


def test_get_column_logical_type(sample_series):
    assert isinstance(
        _get_column_logical_type(sample_series, None, "col_name"),
        Categorical,
    )

    assert isinstance(
        _get_column_logical_type(sample_series, Datetime, "col_name"),
        Datetime,
    )


def test_parse_logical_type():
    assert isinstance(_parse_logical_type("Datetime", "col_name"), Datetime)
    assert isinstance(_parse_logical_type(Datetime, "col_name"), Datetime)

    ymd_format = Datetime(datetime_format="%Y-%m-%d")
    assert _parse_logical_type(ymd_format, "col_name") == ymd_format


def test_parse_logical_type_errors():
    error = "Invalid logical type specified for 'col_name'"
    with pytest.raises(TypeError, match=error):
        _parse_logical_type(int, "col_name")


def test_col_is_datetime():
    inputs = [
        pd.to_datetime(pd.Series(["2020-01-01", "2021-02-02", "2022-03-03"])),
        pd.to_datetime(pd.Series([pd.NA, "2021-02-02", "2022-03-03"])),
        pd.Series([1, 2, 3]),
        pd.Series([pd.NA, 2, 3]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([pd.NA, 2.0, 3.0]),
        pd.Series(["2020-01-01", "2021-02-02", "2022-03-03"]),
        pd.Series([pd.NA, "2021-02-02", "2022-03-03"]),
        pd.Series(["a", "b", "c"]),
        pd.Series([pd.NA, "b", "c"]),
        pd.Series([pd.NA, pd.NA, pd.NA]),
        pd.Series([1950.5, 1951, 1953.2, 1955.3]),
        pd.Series([1950, 1951, 1953, 1955]),
        pd.Series([1950.5, 1951, 1953.2, np.nan]),
        pd.Series([1950, 1951, 1953, np.nan]),
        pd.Series([1950.5, 1951, 1953.2, pd.NA]),
        pd.Series([1950, 1951, 1953, pd.NA]),
        pd.Series([1950, 1951, 1953, None]),
    ]

    expected_values = [
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    for input, expected in list(zip(inputs, expected_values)):
        actual = col_is_datetime(input)
        assert actual is expected


def test_infer_datetime_format(datetimes):
    for series in datetimes:
        fmt = _infer_datetime_format(series)
        assert fmt == "%m/%d/%Y"

    dt = pd.Series(
        ["3/11/2000 9:00", "3/11/2000 10:00", "3/11/2000 11:00", "3/11/2000 12:00"],
    )
    fmt = _infer_datetime_format(dt)
    assert fmt == "%m/%d/%Y %H:%M"

    if version.parse(pd.__version__) >= version.parse("2.0.2"):
        # There was a pandas bug in earlier versions that was casing the
        # hour to get inferred incorrectly as %H (0-24) when it should have
        # been inferred as %I (0-12) when AM/PM is present.
        # Pandas Issue: https://github.com/pandas-dev/pandas/issues/53147
        # https://github.com/alteryx/woodwork/pull/1158
        dt = pd.Series(["Tue 24 Aug 2021 01:30:48 AM"])
        fmt = _infer_datetime_format(dt)
        assert fmt == "%a %d %b %Y %I:%M:%S %p"

        # https://github.com/alteryx/woodwork/pull/1158
        dt = pd.Series(["Tuesday 24 Aug 2021 01:30:48 AM"])
        fmt = _infer_datetime_format(dt)
        assert fmt == "%A %d %b %Y %I:%M:%S %p"

    dt = pd.Series(["Tue 24 Aug 2021 01:30:48"])
    fmt = _infer_datetime_format(dt)
    assert fmt == "%a %d %b %Y %H:%M:%S"

    dt = pd.Series(
        ["13-03-16 10:30:15+0000", "14-03-16 10:30:15+0000", "15-03-16 10:30:15+0000"],
    )
    fmt = _infer_datetime_format(dt)
    assert fmt == "%y-%m-%d %H:%M:%S%z"

    dt = pd.Series(
        [
            pd.Timestamp("13-03-16 10:30:15+0000"),
            pd.Timestamp("14-03-16 10:30:15+0000"),
            pd.Timestamp("15-03-16 10:30:15+0000"),
        ],
    )
    fmt = _infer_datetime_format(dt)
    assert fmt == "%Y-%m-%d %H:%M:%S%z"


def test_infer_datetime_format_all_null():
    missing_data = [
        pd.Series([None, None, None]),
        pd.Series([np.nan, np.nan, np.nan]),
        pd.Series([pd.NA, pd.NA, pd.NA]),
        pd.Series([]),
    ]

    for pd_series in missing_data:
        assert _infer_datetime_format(pd_series) is None
        if dd:
            dd_series = dd.from_pandas(pd_series, npartitions=2)
            assert _infer_datetime_format(dd_series) is None
        if ps:
            ks_series = ps.from_pandas(pd_series)
            assert _infer_datetime_format(ks_series) is None


def test_is_categorical() -> None:
    # not categorical because unhashable type (list)
    assert not _is_categorical_series(pd.Series([[1]]), 0)
    assert not _is_categorical_series(pd.Series([None, [1]]), 0)

    # not categorical because empty series
    assert not _is_categorical_series(pd.Series([]), 0)
    assert not _is_categorical_series(pd.Series([None]), 0)
    assert not _is_categorical_series(pd.Series([None, None]), 0)

    # not categorical because too many unique values
    assert not _is_categorical_series(pd.Series([1, 2]), 0.5)
    assert not _is_categorical_series(pd.Series([1, 2, 3, 1]), 0.5)
    assert not _is_categorical_series(pd.Series([1, 2, 3, 4]), 0.75)

    # categorical
    assert _is_categorical_series(pd.Series([1, 1]), 0.5)
    assert _is_categorical_series(pd.Series([1, 2, 1, 1]), 0.5)
    assert _is_categorical_series(pd.Series([1, 2, 3, 1]), 0.75)


def test_is_latlong_nan():
    assert _is_latlong_nan((np.nan, np.nan))
    assert _is_latlong_nan([np.nan, np.nan])
    assert _is_latlong_nan((np.nan,))
    assert _is_latlong_nan(np.nan)
    assert not _is_latlong_nan((np.nan, 2.0))
    assert not _is_latlong_nan([np.nan, 2.0])
    assert not _is_latlong_nan((2.0, 3.0))
    assert not _is_latlong_nan("test")


def test_callback_caller(mock_callback):
    unit = "yards"
    total = 100
    caller = CallbackCaller(callback=mock_callback, unit=unit, total=total)
    start_time = caller.start_time

    assert mock_callback.total_update == 0
    assert caller.total == total
    assert caller.unit == unit
    assert caller.current_progress == 0
    assert isinstance(start_time, float)

    caller.update(50)
    assert mock_callback.unit == unit
    assert mock_callback.total_update == 50
    assert caller.current_progress == 50
    assert mock_callback.total_elapsed_time > 0
    assert mock_callback.progress_history == [50]


def test_callback_caller_defaults(mock_callback):
    unit = "yards"
    total = 100
    start_time = timer()
    caller = CallbackCaller(
        callback=mock_callback,
        unit=unit,
        total=total,
        start_time=start_time,
        start_progress=20,
    )

    assert caller.start_time == start_time
    assert caller.current_progress == 20


def test_callback_caller_no_callback():
    unit = "yards"
    total = 100
    caller = CallbackCaller(callback=None, unit=unit, total=total)

    assert caller.callback is None
    assert caller.total == 100
    assert caller.current_progress == 0

    caller.update(1)

    assert caller.current_progress == 0


def test_concat_dataframe_or_series_with_series():
    """Tests whether series are correctly concatenated"""
    pandas_series = pd.Series([1, 2, 3])

    assert len(concat_dataframe_or_series(pandas_series, pandas_series)) == 2 * len(
        pandas_series,
    )

    if dd:
        dask_series = dd.from_pandas(pandas_series, npartitions=1)
        assert len(concat_dataframe_or_series(dask_series, dask_series)) == 2 * len(
            dask_series,
        )
    if ps:
        spark_series = ps.Series(data=[1, 2, 3])
        assert len(concat_dataframe_or_series(spark_series, spark_series)) == 2 * len(
            spark_series,
        )


def test_concat_dataframe_or_series_with_series_with_dataframe():
    """Tests whether dataframes are correctly concatenated"""
    d = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data=d)

    assert len(concat_dataframe_or_series(df, df)) == 2 * len(
        df,
    )

    if dd:
        dask_df = dd.from_pandas(df, npartitions=1)
        assert len(concat_dataframe_or_series(dask_df, dask_df)) == 2 * len(
            dask_df,
        )
    if ps:
        spark_df = ps.from_pandas(df)
        assert len(concat_dataframe_or_series(spark_df, spark_df)) == 2 * len(
            spark_df,
        )


def tests_concat_dataframe_or_series_concatenates_in_correct_order():
    """Tests to_add argument is appropriately added to end of base argument"""
    base = pd.Series([1, 2, 3])
    to_add = pd.Series([4, 5, 6])
    concatenated_object = concat_dataframe_or_series(base, to_add)
    assert concatenated_object.head(3).equals(pd.Series([1, 2, 3]))
    assert concatenated_object.tail(3).equals(pd.Series([4, 5, 6]))
