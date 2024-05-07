import sys

import numpy as np
import pandas as pd
import pytest

from woodwork.logical_types import Categorical, CountryCode, Double, Integer, Unknown
from woodwork.type_sys.inference_functions import (
    categorical_func,
    double_func,
    integer_func,
)
from woodwork.type_sys.type_system import INFERENCE_SAMPLE_SIZE, TypeSystem


# Integer Inference Fixtures
@pytest.fixture
def integers():
    return [
        pd.Series(4 * [-1, 2, 1, 7]),
        pd.Series(4 * [-1, 0, 5, 3]),
        pd.Series(4 * [sys.maxsize, -sys.maxsize - 1, 0], dtype="str").astype("int64"),
    ]


# Double Inference Fixtures
@pytest.fixture
def doubles():
    return [
        pd.Series(4 * [-1, 2.5, 1, 7]),
        pd.Series(4 * [1.5, np.nan, 1, 3]),
        pd.Series(4 * [1.5, np.inf, 1, 3]),
        pd.Series(4 * [np.finfo("d").max, np.finfo("d").min, 3, 1]),
    ]


# Boolean Inference Fixtures
@pytest.fixture
def bools():
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


# Datetime Inference Fixtures
@pytest.fixture
def datetimes():
    return [
        pd.Series(["2000-3-11", "2000-3-12", "2000-03-13", "2000-03-14"]),
        pd.Series(["2000-3-11", np.nan, "2000-03-13", "2000-03-14"]),
    ]


# Email Inference Fixtures
@pytest.fixture
def emails():
    return [
        pd.Series(
            ["fl@alteryx.com", "good@email.com", "boaty@mcboatface.com", "foo@bar.com"],
        ),
        pd.Series(["fl@alteryx.com", "good@email.com", "boaty@mcboatface.com", np.nan]),
        pd.Series(
            ["fl@alteryx.com", "good@email.com"] * 2,
        ),
    ]


# Email Inference Fixtures
@pytest.fixture
def bad_emails():
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


# Categorical Inference Fixtures
@pytest.fixture
def categories():
    return [
        pd.Series(10 * ["a", "b", "a", "b"]),
        pd.Series(10 * ["1", "2", "1", "2"]),
        pd.Series(10 * ["a", np.nan, "b", "b"]),
        pd.Series(10 * [1, 2, 1, 2]),
    ]


@pytest.fixture
def categories_dtype():
    return pd.DataFrame(
        {
            "cat": pd.Series(["a", "b", "c", "d"], dtype="category"),
            "non_cat": pd.Series(["a", "b", "c", "d"], dtype="string"),
        },
    )


# Timedelta Inference Fixtures
@pytest.fixture
def timedeltas():
    return [
        pd.Series(pd.to_timedelta(range(4), unit="s")),
        pd.Series([pd.to_timedelta(1, unit="s"), np.nan]),
    ]


# Natural Language Fixtures
@pytest.fixture
def natural_language():
    return [
        pd.Series(
            [
                "Hello World! My name is bob!",
                "I like to move it move it",
                "its cold outside",
            ],
        ),
    ]


# Postal Inference Fixtures
@pytest.fixture
def postal():
    return [
        pd.Series(10 * ["77002", "55106"]),
        pd.Series(10 * ["77002-0000", "55106-0000"]),
        pd.Series(10 * ["12345", "12345", "12345-6789", "12345-0000"]),
    ]


# Unknown Inference Fixtures
@pytest.fixture
def strings():
    return [
        pd.Series(
            ["Mr. John Doe", "Doe, Mrs. Jane", "James Brown", "Ms. Paige Turner"],
        ),
        pd.Series(["y", "no", "N", None]),
        pd.Series(["T", "false", pd.NA, "TRUE"]),
        pd.Series(["1", "1", np.nan, "0.0"]),
    ]


# pd.NA Inference Fixtures
@pytest.fixture
def pdnas():
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


# Empty Series Inference Fixtures
@pytest.fixture
def empty_series():
    return pd.Series([], dtype="object")


# Null Inference Fixtures
@pytest.fixture
def nulls():
    return [
        pd.Series([pd.NA, pd.NA, pd.NA, pd.NA]),
        pd.Series([np.nan, np.nan, np.nan, np.nan]),
        pd.Series([None, None, None, None]),
        pd.Series([None, np.nan, pd.NA, None]),
        pd.Series(["None", "null", "n/a", "NAN"]),
    ]


@pytest.fixture
def large_df():
    df = pd.DataFrame()
    df["int_nullable"] = [int(i) for i in range(INFERENCE_SAMPLE_SIZE)] + [np.nan]
    df["bool_nullable"] = [True, False] * int(INFERENCE_SAMPLE_SIZE // 2) + [pd.NA]
    df["floats"] = [int(i) for i in range(INFERENCE_SAMPLE_SIZE)] + [1.2]
    df["constant"] = [None] * (INFERENCE_SAMPLE_SIZE + 1)
    return df


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
def urls():
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


# Phone Number Inference Fixtures
@pytest.fixture
def phone():
    return [
        pd.Series([f"200.200.786{i}" for i in range(9)]),
        pd.Series(["311-311-3156", "(755) 755 7109", "+1(288)-288-7772"] * 3),
    ]


# IP Address Inference Fixtures
@pytest.fixture
def ip():
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
