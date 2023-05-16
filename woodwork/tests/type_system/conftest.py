import sys

import numpy as np
import pandas as pd
import pytest

from woodwork.logical_types import Categorical, CountryCode, Double, Integer, Unknown
from woodwork.tests.testing_utils import pd_to_dask, pd_to_spark
from woodwork.type_sys.inference_functions import (
    categorical_func,
    double_func,
    integer_func,
)
from woodwork.type_sys.type_system import INFERENCE_SAMPLE_SIZE, TypeSystem


# Integer Inference Fixtures
@pytest.fixture
def pandas_integers():
    return [
        pd.Series(4 * [-1, 2, 1, 7]),
        pd.Series(4 * [-1, 0, 5, 3]),
        pd.Series(4 * [sys.maxsize, -sys.maxsize - 1, 0], dtype="str").astype("int64"),
    ]


@pytest.fixture
def dask_integers(pandas_integers):
    return [pd_to_dask(series) for series in pandas_integers]


@pytest.fixture
def spark_integers(pandas_integers):
    return [pd_to_spark(series) for series in pandas_integers]


@pytest.fixture(params=["pandas_integers", "dask_integers", "spark_integers"])
def integers(request):
    return request.getfixturevalue(request.param)


# Double Inference Fixtures
@pytest.fixture
def pandas_doubles():
    return [
        pd.Series(4 * [-1, 2.5, 1, 7]),
        pd.Series(4 * [1.5, np.nan, 1, 3]),
        pd.Series(4 * [1.5, np.inf, 1, 3]),
        pd.Series(4 * [np.finfo("d").max, np.finfo("d").min, 3, 1]),
    ]


@pytest.fixture
def dask_doubles(pandas_doubles):
    return [pd_to_dask(series) for series in pandas_doubles]


@pytest.fixture
def spark_doubles(pandas_doubles):
    return [pd_to_spark(series) for series in pandas_doubles]


@pytest.fixture(params=["pandas_doubles", "dask_doubles", "spark_doubles"])
def doubles(request):
    return request.getfixturevalue(request.param)


# Boolean Inference Fixtures
@pytest.fixture
def pandas_bools():
    return [
        pd.Series([True, False, True, True]),
        pd.Series([True, np.nan, True, True]),
        pd.Series(["y", "n", "N", "Y"]),
        pd.Series(["True", "false", "FALSE", "TRUE"]),
        pd.Series(["t", "f", "T", "T"]),
        pd.Series(["yes", "no", "NO", "Yes"]),
        pd.Series(["y", "n", "N", None]),
        pd.Series(["True", "false", pd.NA, "TRUE"]),
        pd.Series(["t", "f", "T", np.nan]),
        pd.Series(["yes", "no", "NO", pd.NA]),
    ]


@pytest.fixture
def dask_bools(pandas_bools):
    return [pd_to_dask(series) for series in pandas_bools]


@pytest.fixture
def spark_bools(pandas_bools):
    return [pd_to_spark(series) for series in pandas_bools]


@pytest.fixture(params=["pandas_bools", "dask_bools", "spark_bools"])
def bools(request):
    return request.getfixturevalue(request.param)


# Datetime Inference Fixtures
@pytest.fixture
def pandas_datetimes():
    return [
        pd.Series(["2000-3-11", "2000-3-12", "2000-03-13", "2000-03-14"]),
        pd.Series(["2000-3-11", np.nan, "2000-03-13", "2000-03-14"]),
    ]


@pytest.fixture
def dask_datetimes(pandas_datetimes):
    return [pd_to_dask(series) for series in pandas_datetimes]


@pytest.fixture
def spark_datetimes(pandas_datetimes):
    return [pd_to_spark(series) for series in pandas_datetimes]


@pytest.fixture(params=["pandas_datetimes", "dask_datetimes", "spark_datetimes"])
def datetimes(request):
    return request.getfixturevalue(request.param)


# Email Inference Fixtures
@pytest.fixture
def pandas_emails():
    return [
        pd.Series(
            ["fl@alteryx.com", "good@email.com", "boaty@mcboatface.com", "foo@bar.com"],
        ),
        pd.Series(["fl@alteryx.com", "good@email.com", "boaty@mcboatface.com", np.nan]),
        pd.Series(
            ["fl@alteryx.com", "good@email.com"] * 2,
        ),
    ]


@pytest.fixture
def dask_emails(pandas_emails):
    return [pd_to_dask(series) for series in pandas_emails]


@pytest.fixture
def spark_emails(pandas_emails):
    return [pd_to_spark(series) for series in pandas_emails]


@pytest.fixture(params=["pandas_emails", "dask_emails", "spark_emails"])
def emails(request):
    return request.getfixturevalue(request.param)


# Email Inference Fixtures
@pytest.fixture
def bad_pandas_emails():
    return [
        pd.Series(["fl@alteryx.com", "not_an_email", "good@email.com", "foo@bar.com"]),
        pd.Series(["fl@alteryx.com", "b☃d@email.com", "good@email.com", np.nan]),
        pd.Series(["fl@alteryx.com", "@email.com", "good@email.com", "foo@bar.com"]),
        pd.Series(["fl@alteryx.com", "bad@email", "good@email.com", np.nan]),
        pd.Series([np.nan, np.nan, np.nan, np.nan]),
        pd.Series([1, 2, 3, 4]).astype("int"),
        pd.Series([{"key": "value"}]).astype("O"),
        pd.Series([(1, 2), (3, 4)]).astype("O"),
    ]


@pytest.fixture
def bad_dask_emails(bad_pandas_emails):
    return [pd_to_dask(series) for series in bad_pandas_emails]


@pytest.fixture
def bad_spark_emails(bad_pandas_emails):
    return [pd_to_spark(series) for series in bad_pandas_emails]


@pytest.fixture(params=["bad_pandas_emails", "bad_dask_emails", "bad_spark_emails"])
def bad_emails(request):
    return request.getfixturevalue(request.param)


# Categorical Inference Fixtures
@pytest.fixture
def pandas_categories():
    return [
        pd.Series(10 * ["a", "b", "a", "b"]),
        pd.Series(10 * ["1", "2", "1", "2"]),
        pd.Series(10 * ["a", np.nan, "b", "b"]),
        pd.Series(10 * [1, 2, 1, 2]),
    ]


@pytest.fixture
def dask_categories(pandas_categories):
    return [pd_to_dask(series) for series in pandas_categories]


@pytest.fixture
def spark_categories(pandas_categories):
    return [pd_to_spark(series) for series in pandas_categories]


@pytest.fixture(params=["pandas_categories", "dask_categories", "spark_categories"])
def categories(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pandas_categories_dtype():
    return pd.DataFrame(
        {
            "cat": pd.Series(["a", "b", "c", "d"], dtype="category"),
            "non_cat": pd.Series(["a", "b", "c", "d"], dtype="string"),
        },
    )


@pytest.fixture
def dask_categories_dtype(pandas_categories_dtype):
    return pd_to_dask(pandas_categories_dtype)


@pytest.fixture(params=["pandas_categories_dtype", "dask_categories_dtype"])
def categories_dtype(request):
    # Spark doesn't support the "category" dtype.  We just leave it out for
    # now.
    return request.getfixturevalue(request.param)


# Timedelta Inference Fixtures
@pytest.fixture
def pandas_timedeltas():
    return [
        pd.Series(pd.to_timedelta(range(4), unit="s")),
        pd.Series([pd.to_timedelta(1, unit="s"), np.nan]),
    ]


@pytest.fixture
def dask_timedeltas(pandas_timedeltas):
    return [pd_to_dask(series) for series in pandas_timedeltas]


@pytest.fixture(params=["pandas_timedeltas", "dask_timedeltas"])
def timedeltas(request):
    return request.getfixturevalue(request.param)


# Natural Language Fixtures
@pytest.fixture
def pandas_natural_language():
    return [
        pd.Series(
            [
                "Hello World! My name is bob!",
                "I like to move it move it",
                "its cold outside",
            ],
        ),
    ]


@pytest.fixture
def dask_natural_language(pandas_natural_language):
    return [pd_to_dask(series) for series in pandas_natural_language]


@pytest.fixture
def spark_natural_language(pandas_natural_language):
    return [pd_to_spark(series) for series in pandas_natural_language]


@pytest.fixture(
    params=[
        "pandas_natural_language",
        "dask_natural_language",
        "spark_natural_language",
    ],
)
def natural_language(request):
    return request.getfixturevalue(request.param)


# Postal Inference Fixtures
@pytest.fixture
def pandas_postal_codes():
    return [
        pd.Series(10 * ["77002", "55106"]),
        pd.Series(10 * ["77002-0000", "55106-0000"]),
        pd.Series(10 * ["12345", "12345", "12345-6789", "12345-0000"]),
    ]


@pytest.fixture
def dask_postal_codes(pandas_postal_codes):
    return [pd_to_dask(series) for series in pandas_postal_codes]


@pytest.fixture
def spark_postal_codes(pandas_postal_codes):
    return [pd_to_spark(series) for series in pandas_postal_codes]


@pytest.fixture(
    params=["pandas_postal_codes", "dask_postal_codes", "spark_postal_codes"],
)
def postal(request):
    return request.getfixturevalue(request.param)


# Unknown Inference Fixtures
@pytest.fixture
def pandas_strings():
    return [
        pd.Series(
            ["Mr. John Doe", "Doe, Mrs. Jane", "James Brown", "Ms. Paige Turner"],
        ),
        pd.Series(["y", "no", "N", None]),
        pd.Series(["T", "false", pd.NA, "TRUE"]),
        pd.Series(["1", "1", np.nan, "0.0"]),
    ]


@pytest.fixture
def dask_strings(pandas_strings):
    return [pd_to_dask(series) for series in pandas_strings]


@pytest.fixture
def spark_strings(pandas_strings):
    return [pd_to_spark(series) for series in pandas_strings]


@pytest.fixture(params=["pandas_strings", "dask_strings", "spark_strings"])
def strings(request):
    return request.getfixturevalue(request.param)


# pd.NA Inference Fixtures
@pytest.fixture
def pandas_pdnas():
    return [
        pd.Series(
            [
                "Hello World! My name is bob!",
                pd.NA,
                "I like to move it move it",
                "its cold outside",
            ],
        ),
        pd.Series(["Mr. John Doe", pd.NA, "James Brown", "Ms. Paige Turner"]).astype(
            "string",
        ),
        pd.Series([1, pd.NA, 2, 3]).astype("Int64"),
        pd.Series([True, pd.NA, False, True]).astype("boolean"),
    ]


@pytest.fixture
def dask_pdnas(pandas_pdnas):
    return [pd_to_dask(series) for series in pandas_pdnas]


@pytest.fixture(params=["pandas_pdnas", "dask_pdnas"])
def pdnas(request):
    return request.getfixturevalue(request.param)


# Empty Series Inference Fixtures
@pytest.fixture
def pandas_empty_series():
    return pd.Series([], dtype="object")


@pytest.fixture
def dask_empty_series(pandas_empty_series):
    return pd_to_dask(pandas_empty_series)


@pytest.fixture
def pyspark_empty_series(pandas_empty_series):
    return pd_to_spark(pandas_empty_series)


@pytest.fixture(
    params=["pandas_empty_series", "dask_empty_series", "pyspark_empty_series"],
)
def empty_series(request):
    return request.getfixturevalue(request.param)


# Null Inference Fixtures
@pytest.fixture
def pandas_nulls():
    return [
        pd.Series([pd.NA, pd.NA, pd.NA, pd.NA]),
        pd.Series([np.nan, np.nan, np.nan, np.nan]),
        pd.Series([None, None, None, None]),
        pd.Series([None, np.nan, pd.NA, None]),
        pd.Series(["None", "null", "n/a", "NAN"]),
    ]


@pytest.fixture
def dask_nulls(pandas_nulls):
    return [pd_to_dask(series) for series in pandas_nulls]


@pytest.fixture
def spark_nulls(pandas_nulls):
    return [pd_to_spark(series) for series in pandas_nulls]


@pytest.fixture(params=["pandas_nulls", "dask_nulls", "spark_nulls"])
def nulls(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pandas_large_df():
    df = pd.DataFrame()
    df["int_nullable"] = [int(i) for i in range(INFERENCE_SAMPLE_SIZE)] + [np.nan]
    df["bool_nullable"] = [True, False] * int(INFERENCE_SAMPLE_SIZE // 2) + [pd.NA]
    df["floats"] = [int(i) for i in range(INFERENCE_SAMPLE_SIZE)] + [1.2]
    df["constant"] = [None] * (INFERENCE_SAMPLE_SIZE + 1)
    return df


@pytest.fixture
def dask_large_df(pandas_large_df):
    return pd_to_dask(pandas_large_df)


@pytest.fixture
def spark_large_df(pandas_large_df):
    return pd_to_spark(pandas_large_df)


@pytest.fixture(params=["pandas_large_df", "dask_large_df", "spark_large_df"])
def large_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def default_inference_functions():
    return {
        Double: double_func,
        Integer: integer_func,
        Categorical: categorical_func,
        CountryCode: None,
        Unknown: None,
    }


@pytest.fixture
def default_relationships():
    return [(Double, Integer), (Categorical, CountryCode)]


@pytest.fixture
def type_sys(default_inference_functions, default_relationships):
    return TypeSystem(
        inference_functions=default_inference_functions,
        relationships=default_relationships,
        default_type=Unknown,
    )


# URL Inference Fixtures
@pytest.fixture
def pandas_urls():
    return [
        pd.Series([f"http://url{i}.com" for i in range(100)]),
        pd.Series(
            [
                "http://url.com",
                "http://url.org",
                "https://another.net",
                "https://schoo.edu",
            ]
            * 25,
        ),
    ]


@pytest.fixture
def dask_urls(pandas_urls):
    return [pd_to_dask(series) for series in pandas_urls]


@pytest.fixture
def spark_urls(pandas_urls):
    return [pd_to_spark(series) for series in pandas_urls]


@pytest.fixture(params=["pandas_urls", "dask_urls", "spark_urls"])
def urls(request):
    return request.getfixturevalue(request.param)


# Phone Number Inference Fixtures
@pytest.fixture
def pandas_phone():
    return [
        pd.Series([f"200.200.786{i}" for i in range(9)]),
        pd.Series(["311-311-3156", "(755) 755 7109", "+1(288)-288-7772"] * 3),
    ]


@pytest.fixture
def dask_phone(pandas_phone):
    return [pd_to_dask(series) for series in pandas_phone]


@pytest.fixture
def spark_phone(pandas_phone):
    return [pd_to_spark(series) for series in pandas_phone]


@pytest.fixture(params=["pandas_phone", "dask_phone", "spark_phone"])
def phone(request):
    return request.getfixturevalue(request.param)


# IP Address Inference Fixtures
@pytest.fixture
def pandas_ip():
    return [
        pd.Series(
            [
                "172.16.254.1",
                "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
                "1762:0:0:0:0:B03:1:AF18",
            ]
            * 2,
        ),
        pd.Series([f"172.16.254.{i}" for i in range(6)]),
    ]


@pytest.fixture
def dask_ip(pandas_ip):
    return [pd_to_dask(series) for series in pandas_ip]


@pytest.fixture
def spark_ip(pandas_ip):
    return [pd_to_spark(series) for series in pandas_ip]


@pytest.fixture(params=["pandas_ip", "dask_ip", "spark_ip"])
def ip(request):
    return request.getfixturevalue(request.param)
