from datetime import datetime

import pandas as pd
import pandas.api.types as pdtypes

from data_tables.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    NaturalLanguage,
    Timedelta
)


class DataColumn(object):
    def __init__(self, series, logical_type=None, tags=set()):
        """Create DataColumn

        Args:
            series (pd.Series): Series containing the data associated with the column.
            logical_type (LogicalType, optional): The logical type that should be assigned
                to the column.
            tags (set(str), optional): A set of semantic type tags to assign to the column.
        """
        self.series = series
        self.name = series.name
        if logical_type:
            self.logical_type = logical_type
        else:
            self.logical_type = infer_logical_type(self.series)
        self.dtype = series.dtype
        self.tags = tags


def infer_logical_type(series):
    """Infer variable type for a dataframe column
    Args:
        series (pd.Series): Input Series
    """
    inferred_type = NaturalLanguage

    if series.dtype == 'object':
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

    elif pdtypes.is_numeric_dtype(series.dtype):
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
