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
    Timedelta
)


class DataColumn(object):
    def __init__(self, series, logical_type=None, semantic_types=None):
        """Create DataColumn

        Args:
            series (pd.Series): Series containing the data associated with the column.
            logical_type (LogicalType, optional): The logical type that should be assigned
                to the column. If no value is provided, the LogicalType for the series will
                be inferred.
            semantic_types (dict[str -> dict[str -> str/list]]), optional): A dictionary of semantic
                type tags to assign to the column. Semantic types should be specified as a dictionary
                of dictionaries, where the keys of the outer dictionary represent the semantic type tags,
                and the value is another dictionary of any additional data to store with the semantic type.
                If the additional data is not required, an empty dictionary should be passed. Defaults to
                an empty dictionary, if not specified.
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

        if semantic_types:
            if not isinstance(semantic_types, dict):
                raise TypeError("semantic_types must be a dictionary")
            if not all([isinstance(key, str) for key in semantic_types.keys()]):
                raise TypeError("Semantic types must be specified as strings")
            if not all([isinstance(value, dict) for value in semantic_types.values()]):
                raise TypeError("Additional semantic type data must be specified in a dictionary")
        elif semantic_types is None:
            semantic_types = {}
        self.semantic_types = semantic_types

    def __repr__(self):
        msg = u"<DataColumn: {} ".format(self.name)
        msg += u"(Physical Type = {}) ".format(self.dtype)
        msg += u"(Logical Type = {}) ".format(self.logical_type)
        msg += u"(Semantic Tags = {})>".format(self.tags)
        return msg


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

    elif series.dtype == 'bool':
        inferred_type = Boolean

    elif pdtypes.is_categorical_dtype(series.dtype):
        inferred_type = Categorical

    elif pdtypes.is_integer_dtype(series.dtype):
        inferred_type = Integer

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
