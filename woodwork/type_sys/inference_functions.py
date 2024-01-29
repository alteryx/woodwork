import sys
from typing import Any, Callable, Iterable, Union

import numpy as np
import pandas as pd
from importlib_resources import files
from pandas import CategoricalDtype
from pandas.api import types as pdtypes

import woodwork as ww
from woodwork.config import config
from woodwork.type_sys.utils import _is_categorical_series, col_is_datetime

MAX_INT = sys.maxsize
MIN_INT = -sys.maxsize - 1
Tokens = Iterable[str]

COMMON_WORDS_SET = set(
    word.strip().lower()
    for word in files("woodwork.data").joinpath("1-1000.txt").read_text().split("\n")
    if len(word) > 0
)

NL_delimiters = r"[- \[\].,!\?;\n]"


def categorical_func(series):
    if isinstance(series.dtype, CategoricalDtype):
        return True

    if pdtypes.is_string_dtype(series.dtype) and not col_is_datetime(series):
        categorical_threshold = ww.config.get_option("categorical_threshold")

        return _is_categorical_series(series, categorical_threshold)

    if pdtypes.is_float_dtype(series.dtype) or pdtypes.is_integer_dtype(series.dtype):
        numeric_categorical_threshold = ww.config.get_option(
            "numeric_categorical_threshold",
        )
        if numeric_categorical_threshold is not None:
            return _is_categorical_series(series, numeric_categorical_threshold)
        else:
            return False

    return False


def integer_func(series, is_integer_nullable=None):
    is_int_nullable = (
        is_integer_nullable
        if is_integer_nullable is not None
        else integer_nullable_func(series)
    )

    if is_int_nullable and not series.isnull().any():
        if pdtypes.is_object_dtype(series.dtype):
            return True
        return all(series.mod(1).eq(0))
    return False


def integer_nullable_func(series):
    if pdtypes.is_integer_dtype(series.dtype):
        threshold = ww.config.get_option("numeric_categorical_threshold")
        if threshold is not None:
            return not _is_categorical_series(series, threshold)
        else:
            return True
    elif pdtypes.is_float_dtype(series.dtype):

        def _is_valid_int(value):
            return value >= MIN_INT and value <= MAX_INT and value.is_integer()

        if not series.isnull().any():
            return False
        series_no_null = series.dropna()
        return all(_is_valid_int(v) for v in series_no_null)
    elif pdtypes.is_object_dtype(series.dtype):
        series_no_null = series.dropna()
        try:
            return series_no_null.map(
                lambda x: (isinstance(x, str) and isinstance(int(x), int)),
            ).all()
        except ValueError:
            return False

    return False


def double_func(series):
    if pdtypes.is_float_dtype(series.dtype):
        threshold = ww.config.get_option("numeric_categorical_threshold")
        if threshold is not None:
            return not _is_categorical_series(series, threshold)
        else:
            return True
    elif pdtypes.is_object_dtype(series.dtype):
        series_no_null = series.dropna()
        try:
            # If str and casting to float works, make sure that it isn't just an integer
            return series_no_null.map(
                lambda x: isinstance(x, str) and not float(x).is_integer(),
            ).any()
        except ValueError:
            return False

    return False


def boolean_func(series, is_boolean_nullable=None):
    if is_boolean_nullable is not None:
        if is_boolean_nullable is False:
            return False
        if is_boolean_nullable is True and not series.isnull().any():
            return True
    if not series.isnull().any() and boolean_nullable_func(series):
        return True
    return False


def boolean_nullable_func(series):
    dtype = series.dtype
    if pdtypes.is_bool_dtype(dtype) and not isinstance(dtype, CategoricalDtype):
        return True
    elif pdtypes.is_object_dtype(dtype):
        series_no_null = series.dropna()
        try:
            series_no_null_unq = set(series_no_null)
            if series_no_null_unq.issubset({False, True}):
                return True
            series_lower = set(str(s).lower() for s in series_no_null_unq)
            if series_lower in config.get_option("boolean_inference_strings"):
                return True
        except TypeError:  # Necessary to check for non-hashable values because of object dtype consideration
            return False
    elif pdtypes.is_integer_dtype(dtype) and len(
        config.get_option("boolean_inference_ints"),
    ):
        series_unique = set(series.unique())
        if series_unique == config.get_option("boolean_inference_ints"):
            return True
    return False


def datetime_func(series):
    if col_is_datetime(series):
        return True
    return False


def timedelta_func(series):
    if pdtypes.is_timedelta64_dtype(series.dtype):
        return True
    return False


def num_common_words(wordlist: Union[Tokens, Any]) -> float:
    if not isinstance(wordlist, Iterable):
        return np.nan
    num_common_words = 0
    for x in wordlist:
        if x.lower() in COMMON_WORDS_SET:
            num_common_words += 1
    return num_common_words


def natural_language_func(series):
    tokens = series.astype("string").str.split(NL_delimiters)
    mean_num_common_words = np.nanmean(tokens.map(num_common_words))

    return (
        mean_num_common_words > 1.14
    )  # determined through https://github.com/alteryx/nl_inference


class InferWithRegex:
    def __init__(self, get_regex: Callable[[], str]):
        self.get_regex = get_regex

    def __call__(self, series: pd.Series) -> bool:
        series = series.dropna()
        regex = self.get_regex()

        # Includes a check for object dtypes
        if not (
            isinstance(series.dtype, CategoricalDtype)
            or pdtypes.is_string_dtype(series.dtype)
        ):
            return False

        try:
            series_match_method = series.str.match
        except (AttributeError, TypeError):
            # This can happen either when the inferred dtype for a series is not
            # compatible with the pandas string API (AttributeError) *or* when the
            # inferred dtype is not compatible with the string API `match` method
            # (TypeError)
            return False
        matches = series_match_method(pat=regex)

        return matches.sum() == len(matches)


email_address_func = InferWithRegex(
    lambda: ww.config.get_option("email_inference_regex"),
)
phone_number_func = InferWithRegex(
    lambda: ww.config.get_option("phone_inference_regex"),
)
postal_code_func = InferWithRegex(
    lambda: ww.config.get_option("postal_code_inference_regex"),
)
url_func = InferWithRegex(lambda: ww.config.get_option("url_inference_regex"))
ip_address_func = InferWithRegex(
    lambda: (
        "("
        + ww.config.get_option("ipv4_inference_regex")
        + "|"
        + ww.config.get_option("ipv6_inference_regex")
        + ")"
    ),
)
