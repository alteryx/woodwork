import pandas as pd
import pandas.api.types as pdtypes

import woodwork as ww
from woodwork.type_sys.utils import col_is_datetime

INFERENCE_SAMPLE_SIZE = 10000


def get_inference_sample(series: pd.Series) -> pd.Series:
    f"""
    Return a sample of ``series`` for use during type inference.  If the length
    of ``series`` is less than ``{INFERENCE_SAMPLE_SIZE}``, use the series
    length as the sample size.
    """
    return series.sample(n=min(INFERENCE_SAMPLE_SIZE, len(series)))


def categorical_func(series):
    categorical_threshold = ww.config.get_option('categorical_threshold')
    numeric_categorical_threshold = ww.config.get_option('numeric_categorical_threshold')

    if pdtypes.is_string_dtype(series.dtype) and not col_is_datetime(series):
        # heuristics to predict this some other than categorical
        sample = get_inference_sample(series)
        # catch cases where object dtype cannot be interpreted as a string
        try:
            avg_length = sample.str.len().mean()
            if avg_length > categorical_threshold:
                return False
        except AttributeError:
            pass
        return True

    if pdtypes.is_categorical_dtype(series.dtype):
        return True
    if ((pdtypes.is_float_dtype(series.dtype) or pdtypes.is_integer_dtype(series.dtype)) and
            _is_numeric_categorical(series, numeric_categorical_threshold)):
        return True
    return False


def integer_func(series):
    if integer_nullable_func(series) and not series.isnull().any():
        return True
    return False


def integer_nullable_func(series):
    numeric_categorical_threshold = ww.config.get_option('numeric_categorical_threshold')
    if (pdtypes.is_integer_dtype(series.dtype) and
            not _is_numeric_categorical(series, numeric_categorical_threshold)):
        return True
    return False


def double_func(series):
    numeric_categorical_threshold = ww.config.get_option('numeric_categorical_threshold')
    if (pdtypes.is_float_dtype(series.dtype) and
            not _is_numeric_categorical(series, numeric_categorical_threshold)):
        return True
    return False


def boolean_func(series):
    if boolean_nullable_func(series) and not series.isnull().any():
        return True
    return False


def boolean_nullable_func(series):
    if pdtypes.is_bool_dtype(series.dtype) and not pdtypes.is_categorical_dtype(series.dtype):
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


def email_address_func(series: pd.Series) -> bool:
    regex = ww.config.get_option('email_inference_regex')

    # Includes a check for object dtypes
    if not pdtypes.is_string_dtype(series.dtype):
        return False

    sample = get_inference_sample(series)
    try:
        sample_match_method = sample.str.match
    except (AttributeError, TypeError):
        # This can happen either when the inferred dtype for a series is not
        # compatible with the pandas string API (AttributeError) *or* when the
        # inferred dtype is not compatible with the string API `match` method
        # (TypeError)
        return False
    matches = sample_match_method(pat=regex)

    return matches.sum() == matches.count()


def _is_numeric_categorical(series, threshold):
    return threshold != -1 and series.nunique() < threshold
