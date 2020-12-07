import pandas.api.types as pdtypes

import woodwork as ww
from woodwork.type_sys.utils import col_is_datetime


def categorical_func(series):
    natural_language_threshold = ww.config.get_option('natural_language_threshold')
    numeric_categorical_threshold = ww.config.get_option('numeric_categorical_threshold')

    if pdtypes.is_string_dtype(series.dtype) and not col_is_datetime(series):
        # heuristics to predict this some other than categorical
        sample = series.sample(min(10000, len(series)))
        # catch cases where object dtype cannot be interpreted as a string
        try:
            avg_length = sample.str.len().mean()
            if avg_length > natural_language_threshold:
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
    if pdtypes.is_bool_dtype(series.dtype):
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


def _is_numeric_categorical(series, threshold):
    return threshold != -1 and series.nunique() < threshold
