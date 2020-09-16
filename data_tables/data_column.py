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
    WholeNumber,
    str_to_logical_type
)


class DataColumn(object):
    def __init__(self, series, logical_type=None, semantic_tags=None):
        """Create DataColumn

        Args:
            series (pd.Series): Series containing the data associated with the column.
            logical_type (LogicalType, optional): The logical type that should be assigned
                to the column. If no value is provided, the LogicalType for the series will
                be inferred.
            semantic_tags (str or list, optional): Semantic tags to assign to the column.
                Defaults to an empty set if not specified. There are two options for
                specifying the semantic tags:
                    (str) If only one semantic tag is being set, a single string can be passed.
--> not sure if this should be just a list or maybe list/set???
                    (list) If muliple tags are being set, a list of strings can be passed.
        """
        self.series = series
        self.set_logical_type(logical_type)
        self.set_semantic_tags(semantic_tags)

    def __repr__(self):
        msg = u"<DataColumn: {} ".format(self.name)
        msg += u"(Physical Type = {}) ".format(self.dtype)
        msg += u"(Logical Type = {}) ".format(self.logical_type)
        msg += u"(Semantic Tags = {})>".format(self.semantic_tags)
        return msg

    def set_logical_type(self, logical_type):
        if logical_type:
            if logical_type in LogicalType.__subclasses__():
                self._logical_type = logical_type
            elif isinstance(logical_type, str):
                self._logical_type = str_to_logical_type(logical_type)
            else:
                raise TypeError(f"Invalid logical type specified for '{self.series.name}'")
        else:
            self._logical_type = infer_logical_type(self.series)

    def set_semantic_tags(self, semantic_tags):
        """Replace semantic tags with passed values"""
        self._semantic_tags = _parse_semantic_tags(semantic_tags)

    @property
    def logical_type(self):
        return self._logical_type

    @property
    def semantic_tags(self):
        return self._semantic_tags

    @property
    def name(self):
        return self.series.name

    @property
    def dtype(self):
        return self.series.dtype


def _parse_semantic_tags(semantic_tags):
    if not semantic_tags:
        return set()

    if type(semantic_tags) not in [list, str]:
        raise TypeError("semantic_tags must be a string or list")

    if isinstance(semantic_tags, str):
        return {semantic_tags}

    if not all([isinstance(tag, str) for tag in semantic_tags]):
        raise TypeError("Semantic tags must be specified as strings")

    return set(semantic_tags)


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
