from importlib import resources as pkg_resources
from typing import Any, Callable, Iterable, Union

import numpy as np
import pandas as pd
from pandas.api import types as pdtypes

import woodwork as ww
from woodwork import data
from woodwork.type_sys.utils import (
    _is_categorical_series,
    _is_cudf_series,
    col_is_datetime,
)
from woodwork.utils import import_or_none

Tokens = Iterable[str]

COMMON_WORDS_SET = set(
    line.strip().lower() for line in pkg_resources.open_text(data, "1-1000.txt")
)

NL_delimiters = r"[- \[\].,!\?;\n]"
cudf = import_or_none("cudf")


def categorical_func(series):
    if pdtypes.is_categorical_dtype(series.dtype):
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


def integer_func(series):
    if _is_cudf_series(series):
        return series.mod(1).eq(0).all()
    if integer_nullable_func(series) and not series.isnull().any():
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
        if not series.isnull().any():
            return False
        series_no_null = series.dropna()
        return all(series_no_null.mod(1).eq(0))

    return False


def double_func(series):
    if pdtypes.is_float_dtype(series.dtype):
        threshold = ww.config.get_option("numeric_categorical_threshold")
        if threshold is not None:
            return not _is_categorical_series(series, threshold)
        else:
            return True

    return False


def boolean_func(series):
    if boolean_nullable_func(series) and not series.isnull().any():
        return True
    return False


def boolean_nullable_func(series):
    if pdtypes.is_bool_dtype(series.dtype) and not pdtypes.is_categorical_dtype(
        series.dtype,
    ):
        return True
    elif pdtypes.is_object_dtype(series.dtype):
        series_no_null = series.dropna()
        try:
            series_no_null_unq = set(series_no_null)
            if series_no_null_unq in [
                {False, True},
                {True},
                {False},
            ]:
                return True
        except TypeError:  # Necessary to check for non-hashable values because of object dtype consideration
            return False
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
    if _is_cudf_series(series):
        # tokens.map(num_common_words)
        # *** TypeError: Operation <function num_common_words at 0x7fa0d605ce50> not supported for dtype list.
        # mean_num_common_words = tokens.mean(skipna=True)
        return False
    else:
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
        if not pdtypes.is_string_dtype(series.dtype):
            return False

        try:
            series_match_method = series.str.match
        except (AttributeError, TypeError):
            # This can happen either when the inferred dtype for a series is not
            # compatible with the pandas string API (AttributeError) *or* when the
            # inferred dtype is not compatible with the string API `match` method
            # (TypeError)
            return False
        if _is_cudf_series(series) and self.get_regex() == ww.config.get_option(
            "email_inference_regex",
        ):
            # cudf.series.match does not like default Email Regex
            matches = series_match_method(pat="^\S+@\S+$")
        elif _is_cudf_series(series) and self.get_regex() == ww.config.get_option(
            "phone_inference_regex",
        ):
            matches = series_match_method(
                pat="^\(?([0-9]{3})\)?[-.●]?([0-9]{3})[-.●]?([0-9]{4})$",
            )
        else:
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
