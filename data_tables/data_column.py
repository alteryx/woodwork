from datetime import datetime

import pandas as pd
import pandas.api.types as pdtypes

from data_tables.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    LogicalType,
    NaturalLanguage,
    Timedelta,
    WholeNumber
)


class DataColumn(object):
    def __init__(self, series, logical_type=None, semantic_types=None):
        """Create DataColumn

        Args:
            series (pd.Series): Series containing the data associated with the column.
            logical_type (LogicalType, optional): The logical type that should be assigned
                to the column. If no value is provided, the LogicalType for the series will
                be inferred.
            semantic_types (str/list/dict, optional): Semantic type tags to assign to the column.
                Defaults to an empty dictionary, if not specified. There are several options for
                specifying the semantic types:
                    (str) If no aditional data is needed and only one semantic type is being set,
                    a single string can be passed.

                    (list) If muliple types are being set and none require additional data, a list
                    of strings can be passed.

                    (dict) For semantic types that require additional data, a dictionary should be
                    passed. In this dictionary, the keys should be strings correponding to the type
                    name and the values should be a dictionary containing any additional data, or
                    `None` if no additional data is being set for a particular semantic type.
        """
        self.series = series
        self.name = series.name
        if logical_type:
            if logical_type not in LogicalType.__subclasses__():
                raise TypeError(f"Invalid logical type specified for '{series.name}'")
            self.logical_type = logical_type
        else:
            self.logical_type = infer_logical_type(self.series)
        self.dtype = series.dtype

        self.semantic_types = _parse_semantic_types(semantic_types)

    def __repr__(self):
        msg = u"<DataColumn: {} ".format(self.name)
        msg += u"(Physical Type = {}) ".format(self.dtype)
        msg += u"(Logical Type = {}) ".format(self.logical_type)
        msg += u"(Semantic Tags = {})>".format(self.semantic_types)
        return msg


def _parse_semantic_types(semantic_types):
    if not semantic_types:
        return {}

    if not type(semantic_types) in [dict, list, str]:
        raise TypeError("semantic_types must be a string, list or dictionary")
    if isinstance(semantic_types, list):
        keys = semantic_types
        values = []
    elif isinstance(semantic_types, dict):
        keys = semantic_types.keys()
        values = [value or {} for value in semantic_types.values()]
    else:
        keys = []
        values = []
    if not all([isinstance(key, str) for key in keys]):
        raise TypeError("Semantic types must be specified as strings")
    if not all([isinstance(value, dict) for value in values]):
        raise TypeError("Additional semantic type data must be specified in a dictionary")

    if isinstance(semantic_types, str):
        return {semantic_types: {}}

    if isinstance(semantic_types, list):
        return {key: {} for key in semantic_types}

    return {key: value or {} for key, value in zip(keys, values)}


def infer_logical_type(series):
    """Infer logical type for a dataframe column
    Args:
        series (pd.Series): Input Series
    """
    inferred_type = NaturalLanguage

    if pdtypes.is_string_dtype(series.dtype):
        if col_is_datetime(series):
            inferred_type = Datetime
        else:
            inferred_type = Categorical

            # heuristics to predict this some other than categorical
            sample = series.sample(min(10000, len(series)))

            # catch cases where object dtype cannot be interpreted as a string
            try:
                avg_length = sample.str.len().mean()
                if avg_length > 10:
                    inferred_type = NaturalLanguage
            except AttributeError:
                pass

    elif pdtypes.is_bool_dtype(series.dtype):
        inferred_type = Boolean

    elif pdtypes.is_categorical_dtype(series.dtype):
        inferred_type = Categorical

    elif pdtypes.is_integer_dtype(series.dtype):
        if any(series < 0):
            inferred_type = Integer
        else:
            inferred_type = WholeNumber

    elif pdtypes.is_float_dtype(series.dtype):
        inferred_type = Double

    elif col_is_datetime(series):
        inferred_type = Datetime

    elif pdtypes.is_timedelta64_dtype(series.dtype):
        inferred_type = Timedelta

    return inferred_type


def col_is_datetime(col):
    """Determine if a dataframe column contains datetime values or not. Returns True if column
    contains datetimes, False if not."""
    if (col.dtype.name.find('datetime') > -1 or
            (len(col) and isinstance(col.iloc[0], datetime))):
        return True

    # if it can be casted to numeric, it's not a datetime
    dropped_na = col.dropna()
    try:
        pd.to_numeric(dropped_na, errors='raise')
    except (ValueError, TypeError):
        # finally, try to cast to datetime
        if col.dtype.name.find('str') > -1 or col.dtype.name.find('object') > -1:
            try:
                pd.to_datetime(dropped_na, errors='raise')
            except Exception:
                return False
            else:
                return True

    return False
