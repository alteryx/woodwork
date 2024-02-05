import numpy as np
import pandas as pd
import pytest

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
    PersonFullName,
    PhoneNumber,
    Unknown,
)
from woodwork.tests.testing_utils import pd_to_dask, pd_to_spark
from woodwork.utils import import_or_none


@pytest.fixture(scope="session", autouse=True)
def spark_session():
    pyspark = import_or_none("pyspark.sql")

    if pyspark:
        spark = (
            pyspark.SparkSession.builder.master("local[*]")
            .config(
                "spark.driver.extraJavaOptions",
                "-Dio.netty.tryReflectionSetAccessible=True",
            )
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )

        return spark


@pytest.fixture(params=["sample_df_pandas", "sample_df_dask", "sample_df_spark"])
def sample_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_df_pandas():
    return pd.DataFrame(
        {
            "id": range(4),
            "full_name": [
                "Mr. John Doe",
                "Doe, Mrs. Jane",
                "James Brown",
                "Ms. Paige Turner",
            ],
            "email": [
                "john.smith@example.com",
                np.nan,
                "team@featuretools.com",
                "junk@example.com",
            ],
            "phone_number": [
                "5555555555",
                "555-555-5555",
                "1-(555)-555-5555",
                "555-555-5555",
            ],
            "age": pd.Series([pd.NA, 33, 33, 57], dtype="Int64"),
            "signup_date": [pd.to_datetime("2020-09-01")] * 4,
            "is_registered": pd.Series([pd.NA, False, True, True], dtype="boolean"),
            "double": [0, 1, 0.0001, -5.4321],
            "double_with_nan": [np.nan, -123.45, 0.0001, 1],
            "integer": [-1, 4, 9, 25],
            "nullable_integer": pd.Series([pd.NA, -36, 49, 64], dtype="Int64"),
            "boolean": [True, False, False, True],
            "categorical": pd.Series(["a", "b", "c", "a"], dtype="category"),
            "datetime_with_NaT": [pd.to_datetime("2020-09-01")] * 3 + [pd.NaT],
            "url": [
                "https://github.com/alteryx/woodwork",
                "https://twitter.com/AlteryxOSS",
                np.nan,
                "http://google.com",
            ],
            "ip_address": [
                "172.16.254.1",
                "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
                "1762:0:0:0:0:B03:1:AF18",
                np.nan,
            ],
        },
    )


@pytest.fixture()
def sample_df_dask(sample_df_pandas):
    dask = pytest.importorskip("dask", reason="Dask not installed, skipping")
    dask.config.set({"dataframe.convert-string": False})
    dd = dask.dataframe
    return dd.from_pandas(sample_df_pandas, npartitions=1)


@pytest.fixture()
def sample_df_spark(sample_df_pandas):
    ps = pytest.importorskip(
        "pyspark.pandas",
        reason="Pyspark pandas not installed, skipping",
    )
    return ps.from_pandas(sample_df_pandas)


@pytest.fixture(
    params=[
        "comprehensive_df_pandas",
        "comprehensive_df_dask",
        "comprehensive_df_spark",
    ],
)
def comprehensive_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def comprehensive_df_pandas():
    df = pd.DataFrame()
    df["ints"] = np.random.choice([i for i in range(-100, 100)], 1_000)
    df["ints_str"] = np.random.choice([f"{i}" for i in range(-100, 100)], 1_000)
    df["ints_null"] = np.random.choice(
        [i for i in range(-50, 50)] + [pd.NA, np.nan, None],
        1_000,
    )
    df["ints_null_str"] = np.random.choice(
        [f"{i}" for i in range(-50, 50)] + [pd.NA, np.nan, None],
        1_000,
    )
    df["floats"] = np.random.choice([i * 1.1 for i in range(-100, 100)], 1_000)
    df["floats_str"] = np.random.choice([f"{i * 1.1}" for i in range(-100, 100)], 1_000)
    df["floats_null"] = np.random.choice(
        [i * 1.1 for i in range(-50, 50)] + [pd.NA, np.nan, None],
        1_000,
    )
    df["floats_null_str"] = np.random.choice(
        [f"{i * 1.1}" for i in range(-50, 50)] + [pd.NA, np.nan, None],
        1_000,
    )
    df["int_float_mixed"] = np.random.choice(
        [f"{i}" for i in range(-50, 50)] + ["3.14"],
        1_000,
    )
    df["int_float_mixed_null"] = np.random.choice(
        [f"{i}" for i in range(-50, 50)] + ["3.14", pd.NA, np.nan, None],
        1_000,
    )
    df["bools"] = np.random.choice([True, False], 1_000)
    df["bools_str"] = np.random.choice(["y", "n"], 1_000)
    df["bools_null"] = np.random.choice([True, False, pd.NA], 1_000)
    df["bools_null_str"] = np.random.choice(["y", "n", pd.NA], 1_000)
    df["datetimes"] = pd.date_range("01/01/1995", freq="3D", periods=1_000)
    df["datetimes_str"] = [
        "01-05-12",
        "01-11-04",
        "03-21-11",
        "11-01-19",
        "12-28-01",
    ] * 200
    df["datetimes_null_str"] = [
        "01-05-12",
        "01-11-04",
        "03-21-11",
        "11-01-19",
        "12-28-01",
        "04-21-15",
        "06-20-98",
        "10-09-99",
        "01-03-00",
        pd.NA,
    ] * 100
    return df


@pytest.fixture()
def comprehensive_df_dask(comprehensive_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(comprehensive_df_pandas, npartitions=1)


@pytest.fixture()
def comprehensive_df_spark(comprehensive_df_pandas):
    ps = pytest.importorskip(
        "pyspark.pandas",
        reason="Pyspark pandas not installed, skipping",
    )
    return ps.from_pandas(comprehensive_df_pandas)


@pytest.fixture()
def sample_df_phone_numbers():
    return pd.DataFrame(
        {
            "phone_number": [
                "2002007865",
                "311-311-3156",
                "422.422.0461",
                "533 533 8122",
                "644●644●7865",
                "(755) 755 7109",
                "(866) 866-0182",
                "+1 (977) 977 1779",
                "+1(288)-288-7772",
                "+1 399 399 2933",
                "+1-400-400-1109",
                "+1.511.511.2289",
                "+16226229222",
                "17336330019",
                "1.844.744.1842",
                "1●955●855●9272",
                "1 266 966 2283",
                "+001 236 248 8482",
            ],
        },
    )


@pytest.fixture()
def sample_df_postal_code():
    return pd.DataFrame(
        {
            "postal_code": [
                "20020",
                "07865",
                "12345-6789",
                "11111-1111",
            ],
        },
    )


@pytest.fixture()
def datetime_freqs_df_pandas():
    return pd.DataFrame(
        {
            "2D_freq": pd.date_range(start="2020-01-01", end="2020-01-20", freq="2D"),
            "3M_freq": pd.date_range(start="2015-01-01", freq="3M", periods=10),
            "3B_no_freq": pd.date_range(start="2015-01-01", freq="3B", periods=10),
            "1d_skipped_one_freq": pd.date_range(
                start="2020-01-01",
                end="2020-01-11",
                freq="1D",
            ).drop("2020-01-04"),
            "3M_one_nan": list(
                pd.date_range(start="2015-01-01", freq="3M", periods=10).drop(
                    "2015-07-31",
                ),
            )
            + [None],
            "2d_timedelta": pd.date_range(
                start="2020-01-01",
                end="2020-01-20",
                freq="2D",
            )
            - pd.Timestamp("2020-01-01"),
            "ints": range(10),
            "same_date": ["2015-01-01"] * 10,
        },
    )


@pytest.fixture()
def datetime_different_formats():
    formats = [
        "%m/%d/%y",
        "%y/%m/%d",
        "%m/%d/%y %H:%M:%S",
        "%y/%m/%d %H:%M:%S",
        "%m/%d/%y %H:%M:%S%z",
        "%y/%m/%d %H:%M:%S%z",
        "%d/%m/%y",
        "%y/%d/%m",
        "%d/%m/%y %H:%M:%S",
        "%y/%d/%m %H:%M:%S",
        "%d/%m/%y %H:%M:%S%z",
        "%y/%d/%m %H:%M:%S%z",
    ]
    dates = [
        "01/15/24",
        "24/01/15",
        "01/15/24 04:23:45",
        "24/01/15 04:23:45",
        "01/15/24 04:23:45+0000",
        "24/01/15 04:23:45+0000",
        "15/01/24",
        "24/15/01",
        "15/01/24 04:23:45",
        "24/15/01 04:23:45",
        "15/01/24 04:23:45+0000",
        "24/15/01 04:23:45+0000",
    ]
    return [(format_, date_) for format_, date_ in zip(formats, dates)]


@pytest.fixture(
    params=[
        "sample_unsorted_df_pandas",
        "sample_unsorted_df_dask",
        "sample_unsorted_df_spark",
    ],
)
def sample_unsorted_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_unsorted_df_pandas():
    return pd.DataFrame(
        {
            "id": [3, 1, 2, 0],
            "full_name": [
                "Mr. John Doe",
                "Doe, Mrs. Jane",
                "James Brown",
                "Ms. Paige Turner",
            ],
            "email": [
                "john.smith@example.com",
                np.nan,
                "team@featuretools.com",
                "junk@example.com",
            ],
            "phone_number": [
                "5555555555",
                "555-555-5555",
                "1-(555)-555-5555",
                "555-555-5555",
            ],
            "age": [33, 25, 33, 57],
            "signup_date": pd.to_datetime(
                ["2020-09-01", "2020-08-01", "2020-08-02", "2020-09-01"],
            ),
            "is_registered": [True, False, True, True],
        },
    )


@pytest.fixture()
def sample_unsorted_df_dask(sample_unsorted_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(sample_unsorted_df_pandas, npartitions=1)


@pytest.fixture()
def sample_unsorted_df_spark(sample_unsorted_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(sample_unsorted_df_pandas)


@pytest.fixture(
    params=["sample_series_pandas", "sample_series_dask", "sample_series_spark"],
)
def sample_series(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_series_pandas():
    return pd.Series(
        ["a", "b", "c"] + 10 * ["a", "a", "a"],
        name="sample_series",
    ).astype("category")


@pytest.fixture()
def sample_series_dask(sample_series_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(sample_series_pandas, npartitions=1)


@pytest.fixture()
def sample_series_spark(sample_series_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(sample_series_pandas.astype("string"))


@pytest.fixture(
    params=[
        "sample_datetime_series_pandas",
        "sample_datetime_series_dask",
        "sample_datetime_series_spark",
    ],
)
def sample_datetime_series(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_datetime_series_pandas():
    return pd.Series(
        [pd.to_datetime("2020-09-01")] * 4,
        name="sample_datetime_series",
    ).astype("object")


@pytest.fixture()
def sample_datetime_series_dask(sample_datetime_series_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(sample_datetime_series_pandas, npartitions=1)


@pytest.fixture()
def sample_datetime_series_spark(sample_datetime_series_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(sample_datetime_series_pandas.astype(str))


@pytest.fixture()
def ordinal_transform_series_pandas():
    return pd.Series([1, 2, 3], dtype="int64")


@pytest.fixture()
def ordinal_transform_series_dask(ordinal_transform_series_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(ordinal_transform_series_pandas, npartitions=1)


@pytest.fixture()
def ordinal_transform_series_spark(ordinal_transform_series_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(ordinal_transform_series_pandas)


@pytest.fixture()
def time_index_df_pandas():
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "times": ["2019-01-01", "2019-01-02", "2019-01-03", pd.NA],
            "ints": [1, 2, 3, 4],
            "strs": ["1", "2", "3", "4"],
            "letters": ["a", "b", "c", "d"],
            "bools": [True, False, False, True],
        },
    )


@pytest.fixture()
def time_index_df_dask(time_index_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(time_index_df_pandas, npartitions=1)


@pytest.fixture()
def time_index_df_spark(time_index_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(time_index_df_pandas)


@pytest.fixture(
    params=["time_index_df_pandas", "time_index_df_dask", "time_index_df_spark"],
)
def time_index_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def numeric_time_index_df_pandas():
    return pd.DataFrame(
        {
            "floats": pd.Series([1, 2, 3, 4], dtype="float"),
            "ints": pd.Series([1, -2, 3, 4], dtype="int64"),
            "with_null": pd.Series([1, pd.NA, 3, 4], dtype="Int64"),
        },
    )


@pytest.fixture()
def numeric_time_index_df_dask(numeric_time_index_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(numeric_time_index_df_pandas, npartitions=1)


@pytest.fixture()
def numeric_time_index_df_spark(numeric_time_index_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(numeric_time_index_df_pandas)


@pytest.fixture(
    params=[
        "numeric_time_index_df_pandas",
        "numeric_time_index_df_dask",
        "numeric_time_index_df_spark",
    ],
)
def numeric_time_index_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def describe_df_pandas():
    index_data = [0, 1, 2, 3, 4, 5, 6, 7]
    boolean_data = [True, False, True, True, False, True, False, None]
    category_data = ["red", "blue", "red", np.nan, "red", "blue", "red", "yellow"]
    datetime_data = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-02-01 00:00",
            "2020-01-01 08:00",
            "2020-02-02 16:00",
            "2020-02-02 18:00",
            pd.NaT,
            "2020-02-01 00:00",
            "2020-01-02 00:00",
        ],
    )
    formatted_datetime_data = pd.Series(
        [
            "2020~01~01",
            "2020~02~01",
            "2020~03~01",
            "2020~02~02",
            "2020~03~02",
            pd.NaT,
            "2020~02~01",
            "2020~01~02",
        ],
    )
    numeric_data = pd.Series([10, 20, 17, 32, np.nan, 1, 56, 10])
    natural_language_data = [
        "This is a natural language sentence",
        "Duplicate sentence.",
        "This line has numbers in it 000123.",
        "How about some symbols?!",
        "This entry contains two sentences. Second sentence.",
        "Duplicate sentence.",
        np.nan,
        "I am the last line",
    ]
    latlong_data = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (0, 0),
        (np.nan, np.nan),
        (np.nan, 6),
        np.nan,
    ]
    timedelta_data = datetime_data - pd.Timestamp("2020-01-01")
    unknown_data = [
        "unknown1",
        "unknown5",
        "unknown3",
        np.nan,
        "unknown3",
        np.nan,
        "unknown5",
        "unknown5",
    ]

    return pd.DataFrame(
        {
            "index_col": index_data,
            "boolean_col": boolean_data,
            "category_col": category_data,
            "datetime_col": datetime_data,
            "formatted_datetime_col": formatted_datetime_data,
            "numeric_col": numeric_data,
            "natural_language_col": natural_language_data,
            "timedelta_col": timedelta_data,
            "latlong_col": latlong_data,
            "unknown_col": unknown_data,
        },
    )


@pytest.fixture()
def describe_df_dask(describe_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(describe_df_pandas, npartitions=1)


@pytest.fixture()
def describe_df_spark(describe_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(
        describe_df_pandas.applymap(
            lambda tup: [None if pd.isnull(elt) else elt for elt in tup]
            if isinstance(tup, tuple)
            else tup,
        ).drop(columns="timedelta_col"),
    )


@pytest.fixture(params=["describe_df_pandas", "describe_df_dask", "describe_df_spark"])
def describe_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def df_same_mi_pandas():
    return pd.DataFrame(
        {
            "ints": pd.Series([2, pd.NA, 5, 2], dtype="Int64"),
            "floats": pd.Series([1, None, 100, 1]),
            "nans": pd.Series([None, None, None, None]),
            "nat_lang": pd.Series(
                [
                    "this is a very long sentence inferred as a string",
                    None,
                    "test",
                    "test",
                ],
            ),
        },
    )


@pytest.fixture()
def df_same_mi_dask(df_same_mi_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(df_same_mi_pandas, npartitions=1)


@pytest.fixture()
def df_same_mi_spark(df_same_mi_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    df_same_mi_pandas["ints"] = df_same_mi_pandas["ints"].astype("float")
    df_same_mi_pandas["nans"] = df_same_mi_pandas["nans"].astype("float")
    return ps.from_pandas(df_same_mi_pandas)


@pytest.fixture(params=["df_same_mi_pandas", "df_same_mi_dask", "df_same_mi_spark"])
def df_same_mi(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def df_mi_pandas():
    df = pd.DataFrame(
        {
            "ints": pd.Series([1, 2, 1]),
            "ints2": pd.Series([2, 2, 2]),
            "bools": pd.Series([True, False, True]),
            "strs2": pd.Series(["bye", "hi", "bye"]),
            "strs": pd.Series(["hi", "hi", "hi"]),
            "dates": pd.Series(["2020-01-01", "2020-01-01", "1997-01-04"]),
        },
    )
    df = df.loc[df.index.repeat(4)].reset_index(drop=True)
    return df


@pytest.fixture()
def df_mi_dask(df_mi_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(df_mi_pandas, npartitions=1)


@pytest.fixture()
def df_mi_spark(df_mi_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(df_mi_pandas)


@pytest.fixture(params=["df_mi_pandas", "df_mi_dask", "df_mi_spark"])
def df_mi(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def df_mi_unique_pandas():
    df = pd.DataFrame(
        {
            "unique": pd.Series(["hi", "bye", "hello", "goodbye"]),
            "unique_with_one_nan": pd.Series(["hi", "bye", None, "goodbye"]),
            "unique_with_nans": pd.Series([1, None, None, 2]),
            "ints": pd.Series([1, 2, 1, 2]),
        },
    )
    df = df.loc[df.index.repeat(5)].reset_index(drop=True)
    return df


@pytest.fixture()
def df_mi_unique_dask(df_mi_unique_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(df_mi_unique_pandas, npartitions=1)


@pytest.fixture()
def df_mi_unique_spark(df_mi_unique_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(df_mi_unique_pandas)


@pytest.fixture(
    params=["df_mi_unique_pandas", "df_mi_unique_dask", "df_mi_unique_spark"],
)
def df_mi_unique(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def categorical_df_pandas():
    return pd.DataFrame(
        {
            "ints": pd.Series([1, 2, 3, 2]),
            "categories1": pd.Series([1, 100, 1, 100, 200, 200, 200, 200, 3, 100]),
            "bools": pd.Series([True, False, True, False]),
            "categories2": pd.Series(["test", "test", "test2", "test"]),
            "categories3": pd.Series(["test", "test", "test", np.nan]),
        },
    )


@pytest.fixture()
def categorical_df_dask(categorical_df_pandas):
    dask = pytest.importorskip("dask", reason="Dask not installed, skipping")
    dask.config.set({"dataframe.convert-string": False})
    dd = dask.dataframe
    return dd.from_pandas(categorical_df_pandas, npartitions=1)


@pytest.fixture()
def categorical_df_spark(categorical_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(categorical_df_pandas)


@pytest.fixture(
    params=["categorical_df_pandas", "categorical_df_dask", "categorical_df_spark"],
)
def categorical_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def empty_df_pandas():
    return pd.DataFrame({})


@pytest.fixture()
def empty_df_dask(empty_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(empty_df_pandas, npartitions=1)


# Cannot have an empty Spark DataFrame
@pytest.fixture(params=["empty_df_pandas", "empty_df_dask"])
def empty_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def small_df_pandas():
    return pd.DataFrame(
        pd.Series(
            [pd.to_datetime("2020-09-01")] * 4,
            name="sample_datetime_series",
        ).astype("object"),
    )


@pytest.fixture()
def small_df_dask(small_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(small_df_pandas, npartitions=1)


@pytest.fixture()
def small_df_spark(small_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    df = small_df_pandas.astype("datetime64[ns]")
    return ps.from_pandas(df)


@pytest.fixture(params=["small_df_pandas", "small_df_dask", "small_df_spark"])
def small_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def latlong_df_pandas():
    return pd.DataFrame(
        {
            "tuple_ints": pd.Series([(1, 2), (3, 4)]),
            "tuple_strings": pd.Series([("1", "2"), ("3", "4")]),
            "string_tuple": pd.Series(["(1, 2)", "(3, 4)"]),
            "bracketless_string_tuple": pd.Series(["1, 2", "3, 4"]),
            "list_strings": pd.Series([["1", "2"], ["3", "4"]]),
            "combo_tuple_types": pd.Series(["[1, 2]", "(3, 4)"]),
            "null_value": pd.Series([np.nan, (3, 4)]),
            "null_latitude": pd.Series([(np.nan, 2.0), (3.0, 4.0)]),
            "both_null": pd.Series([(np.nan, np.nan), (3.0, 4.0)]),
        },
    )


@pytest.fixture
def latlong_df_dask(latlong_df_pandas):
    dask = pytest.importorskip("dask", reason="Dask not installed, skipping")
    dask.config.set({"dataframe.convert-string": False})
    dd = dask.dataframe
    return dd.from_pandas(latlong_df_pandas, npartitions=1)


@pytest.fixture
def latlong_df_spark(latlong_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(
        latlong_df_pandas.applymap(
            lambda tup: list(tup) if isinstance(tup, tuple) else tup,
        ),
    )


@pytest.fixture(params=["latlong_df_pandas", "latlong_df_dask", "latlong_df_spark"])
def latlong_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def empty_latlong_df_pandas():
    return pd.DataFrame({"latlong": []}, dtype="object")


@pytest.fixture()
def empty_latlong_df_dask(empty_latlong_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(empty_latlong_df_pandas, npartitions=1)


@pytest.fixture()
def empty_latlong_df_spark(empty_latlong_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(
        empty_latlong_df_pandas.applymap(
            lambda tup: list(tup) if isinstance(tup, tuple) else tup,
        ),
    )


@pytest.fixture(
    params=[
        "empty_latlong_df_pandas",
        "empty_latlong_df_dask",
        "empty_latlong_df_spark",
    ],
)
def empty_latlong_df(request):
    return request.getfixturevalue(request.param)


# LatLong Fixtures for testing access to latlong values
@pytest.fixture
def pandas_latlongs():
    return [
        pd.Series([(1.0, 2.0), (3.0, 4.0)]),
        pd.Series([("1", "2"), ("3", "4")]),
        pd.Series([["1", "2"], ["3", "4"]]),
        pd.Series([(1, 2), (3, 4)]),
        pd.Series([[1, 2], [3, 4]]),
        pd.Series(["(1, 2)", "(3, 4)"]),
        pd.Series(["1, 2", "3, 4"]),
        pd.Series(["[1, 2]", "[3, 4]"]),
    ]


@pytest.fixture
def dask_latlongs(pandas_latlongs):
    return [pd_to_dask(series) for series in pandas_latlongs]


@pytest.fixture
def spark_latlongs(pandas_latlongs):
    return [
        pd_to_spark(
            series.apply(lambda tup: list(tup) if isinstance(tup, tuple) else tup),
        )
        for series in pandas_latlongs
    ]


@pytest.fixture(params=["pandas_latlongs", "dask_latlongs", "spark_latlongs"])
def latlongs(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def whitespace_df_pandas():
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "comments": [
                "\nleading newline",
                "trailing newline\n",
                "\n",
                "newline in \n the middle",
                "    leading whitespace",
                "trailing whitespace ",
            ],
        },
    )


@pytest.fixture()
def whitespace_df_dask(whitespace_df_pandas):
    dask = pytest.importorskip("dask", reason="Dask not installed, skipping")
    dask.config.set({"dataframe.convert-string": False})
    dd = dask.dataframe
    return dd.from_pandas(whitespace_df_pandas, npartitions=1)


@pytest.fixture()
def whitespace_df_spark(whitespace_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(whitespace_df_pandas)


@pytest.fixture(
    params=["whitespace_df_pandas", "whitespace_df_dask", "whitespace_df_spark"],
)
def whitespace_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def falsy_names_df_pandas():
    return pd.DataFrame(
        {
            0: ["a", "b", "c"],
            "": [1, 2, 3],
        },
    )


@pytest.fixture()
def falsy_names_df_dask(falsy_names_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(falsy_names_df_pandas, npartitions=1)


@pytest.fixture()
def falsy_names_df_spark(falsy_names_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(
        falsy_names_df_pandas.applymap(
            lambda tup: list(tup) if isinstance(tup, tuple) else tup,
        ),
    )


@pytest.fixture(
    params=["falsy_names_df_pandas", "falsy_names_df_dask", "falsy_names_df_spark"],
)
def falsy_names_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_column_names(sample_df_pandas):
    return sample_df_pandas.columns.to_list()


@pytest.fixture()
def sample_inferred_logical_types():
    return {
        "id": Integer,
        "full_name": Unknown,
        "email": EmailAddress,
        "phone_number": PhoneNumber,
        "age": IntegerNullable,
        "signup_date": Datetime,
        "is_registered": BooleanNullable,
        "double": Double,
        "double_with_nan": Double,
        "integer": Integer,
        "nullable_integer": IntegerNullable,
        "boolean": Boolean,
        "categorical": Categorical,
        "datetime_with_NaT": Datetime,
        "url": URL,
        "ip_address": IPAddress,
    }


@pytest.fixture()
def sample_correct_logical_types():
    return {
        "id": Integer,
        "full_name": PersonFullName,
        "email": EmailAddress,
        "phone_number": PhoneNumber,
        "age": IntegerNullable,
        "signup_date": Datetime,
        "is_registered": BooleanNullable,
        "double": Double,
        "double_with_nan": Double,
        "integer": Integer,
        "nullable_integer": IntegerNullable,
        "boolean": Boolean,
        "categorical": Categorical,
        "datetime_with_NaT": Datetime,
        "url": URL,
        "ip_address": IPAddress,
    }


@pytest.fixture()
def serialize_df_pandas():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "cat_int": [1, 2, 1],
            "ord_int": [1, 2, 1],
            "cat_float": [1.0, 2.0, 1.0],
            "ord_float": [1.0, 2.0, 1.0],
            "cat_bool": [True, False, True],
            "ord_bool": [True, False, True],
        },
    )
    return df


@pytest.fixture()
def serialize_df_dask(serialize_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(serialize_df_pandas, npartitions=1)


@pytest.fixture()
def serialize_df_spark(serialize_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(serialize_df_pandas)


@pytest.fixture(
    params=["serialize_df_pandas", "serialize_df_dask", "serialize_df_spark"],
)
def serialize_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def pandas_datetimes():
    return [
        pd.Series(["3/11/2000", "3/12/2000", "3/13/2000", "3/14/2000"]),
        pd.Series(["3/11/2000", np.nan, "3/13/2000", "3/14/2000"]),
    ]


@pytest.fixture()
def dask_datetimes(pandas_datetimes):
    return [pd_to_dask(series) for series in pandas_datetimes]


@pytest.fixture()
def spark_datetimes(pandas_datetimes):
    return [pd_to_spark(series) for series in pandas_datetimes]


@pytest.fixture(params=["pandas_datetimes", "dask_datetimes", "spark_datetimes"])
def datetimes(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def outliers_df_pandas():
    return pd.DataFrame(
        {
            "has_outliers": [93, 42, 37, -16, 49, 42, 36, 57, 60, 23],
            "no_outliers": [60, 42, 37, 23, 49, 42, 36, 57, 60, 23.0],
            "non_numeric": ["a"] * 10,
            "has_outliers_with_nans": [None, 42, 37, -16, 49, 93, 36, 57, 60, 23],
            "nans": pd.Series([None] * 10, dtype="float64"),
        },
    )


@pytest.fixture()
def outliers_df_dask(outliers_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(outliers_df_pandas, npartitions=1)


@pytest.fixture()
def outliers_df_spark(outliers_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(outliers_df_pandas)


@pytest.fixture(params=["outliers_df_pandas", "outliers_df_dask", "outliers_df_spark"])
def outliers_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def skewed_outliers_df_pandas():
    outliers_df = pd.DataFrame(
        {
            "right_skewed_outliers": [1] * 2
            + [2] * 6
            + [3] * 20
            + [4] * 12
            + [5] * 8
            + [6] * 5
            + [7] * 3
            + [8, 8, 9, 9, 10, 11, 13, 14, 16, 30],
            "right_skewed_outliers_nullable_int": pd.Series(
                [1] * 2
                + [2] * 6
                + [3] * 20
                + [4] * 12
                + [5] * 8
                + [6] * 5
                + [7] * 3
                + [8, 8, 9, 9, 10, 11, 13, 14, 16, 30],
                dtype="Int64",
            ),
            "no_outliers": [60, 42, 37, 23, 49, 42, 36, 57, 60, 23.0] * 6
            + [35, 54, 43, 47, 41, 39],
            "non_numeric": ["a"] * 66,
            "has_outliers_with_nans": [1] * 2
            + [2] * 6
            + [3] * 20
            + [4] * 12
            + [5] * 8
            + [6] * 5
            + [7] * 3
            + [8, None, 9, 9, 10, 11, None, 14, 16, 30],
            "nans": pd.Series([None] * 66, dtype="float64"),
        },
    )
    outliers_df["left_skewed_outliers"] = 31 - outliers_df["right_skewed_outliers"]
    return outliers_df


@pytest.fixture()
def skewed_outliers_df_dask(skewed_outliers_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(skewed_outliers_df_pandas, npartitions=2)


@pytest.fixture()
def skewed_outliers_df_spark(skewed_outliers_df_pandas):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(skewed_outliers_df_pandas)


@pytest.fixture(
    params=[
        "skewed_outliers_df_pandas",
        "skewed_outliers_df_dask",
        "skewed_outliers_df_spark",
    ],
)
def skewed_outliers_df(request):
    return request.getfixturevalue(request.param)


class MockCallback:
    def __init__(self):
        self.progress_history = []
        self.total_update = 0
        self.total_elapsed_time = 0

    def __call__(self, update, progress, total, unit, time_elapsed):
        self.total_update += update
        self.total = total
        self.progress_history.append(progress)
        self.unit = unit
        self.total_elapsed_time = time_elapsed


@pytest.fixture()
def mock_callback():
    return MockCallback()


class MockResultCallback:
    def __init__(self):
        self.results_so_far = []
        self.most_recent_calculation = []

    def __call__(self, results_so_far, most_recent_calculations):
        self.results_so_far.append(results_so_far)
        self.most_recent_calculation.append(most_recent_calculations)


@pytest.fixture()
def mock_results_callback():
    return MockResultCallback()


@pytest.fixture()
def timezones_df_pandas():
    dtypes = {
        "default_1": "datetime64[ns]",
        "utc_1": "datetime64[ns, UTC]",
        "eastern_1": "datetime64[ns, US/Eastern]",
        "default_2": "object",
        "utc_2": "object",
        "eastern_2": "object",
    }
    return pd.DataFrame(
        data=[
            [
                "2022-01-01",
                "2022-01-01 00:00:00+00:00",
                "2022-01-01 00:00:00-05:00",
                "2022-01-01",
                "2022-01-01T00:00:00+00:00",
                "2022-01-01 00:00:00-05:00",
            ],
            [
                "2022-01-02",
                "2022-01-02 00:00:00+00:00",
                "2022-01-02 00:00:00-05:00",
                "2022-01-02",
                "2022-01-02T00:00:00+00:00",
                "2022-01-02 00:00:00-05:00",
            ],
            ["NaT", "NaT", "NaT", "NaT", "NaT", "NaT"],
        ],
        columns=["default_1", "utc_1", "eastern_1", "default_2", "utc_2", "eastern_2"],
    ).astype(dtypes)


@pytest.fixture()
def timezones_df_dask(timezones_df_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(timezones_df_pandas, npartitions=1)


@pytest.fixture(params=["timezones_df_pandas", "timezones_df_dask"])
def timezones_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def postal_code_numeric_series_pandas():
    return pd.Series([77449.0, 11368.0, np.nan, 60629.0, 79936.0, 1234567890.0])


@pytest.fixture()
def postal_code_numeric_series_dask(postal_code_numeric_series_pandas):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(postal_code_numeric_series_pandas, npartitions=1)


@pytest.fixture()
def postal_code_numeric_series_spark(postal_code_numeric_series_pandas):
    ps = pytest.importorskip(
        "pyspark.pandas",
        reason="Pyspark pandas not installed, skipping",
    )
    return ps.from_pandas(postal_code_numeric_series_pandas)


@pytest.fixture(
    params=[
        "postal_code_numeric_series_pandas",
        "postal_code_numeric_series_dask",
        "postal_code_numeric_series_spark",
    ],
)
def postal_code_numeric_series(request):
    return request.getfixturevalue(request.param)
