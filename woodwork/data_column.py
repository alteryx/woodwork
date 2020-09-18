import warnings
from datetime import datetime

import pandas as pd
import pandas.api.types as pdtypes

from woodwork.logical_types import (
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
from woodwork.utils import _parse_semantic_tags


class DataColumn(object):
    def __init__(self, series,
                 logical_type=None,
                 semantic_tags=None,
                 add_standard_tags=True):
        """Create DataColumn

        Args:
            series (pd.Series): Series containing the data associated with the column.
            logical_type (LogicalType, optional): The logical type that should be assigned
                to the column. If no value is provided, the LogicalType for the series will
                be inferred.
            semantic_tags (str or list or set, optional): Semantic tags to assign to the column.
                Defaults to an empty set if not specified. There are two options for
                specifying the semantic tags:
                    (str) If only one semantic tag is being set, a single string can be passed.
                    (list or set) If muliple tags are being set, a list or set of strings can be passed.
            add_standard_tags (bool, optional): If True, will add standard semantic tags to columns based
                on the inferred or specified logical type for the column. Defaults to True.
        """
        self.series = series
        self.add_standard_tags = add_standard_tags
        self._logical_type = self._parse_logical_type(logical_type)
        self.set_semantic_tags(semantic_tags)

    def __repr__(self):
        msg = u"<DataColumn: {} ".format(self.name)
        msg += u"(Physical Type = {}) ".format(self.dtype)
        msg += u"(Logical Type = {}) ".format(self.logical_type)
        msg += u"(Semantic Tags = {})>".format(self.semantic_tags)
        return msg

    def set_logical_type(self, logical_type):
        new_logical_type = self._parse_logical_type(logical_type)
        return DataColumn(series=self.series,
                          logical_type=new_logical_type,
                          add_standard_tags=self.add_standard_tags)

    def _parse_logical_type(self, logical_type):
        if logical_type:
            if logical_type in LogicalType.__subclasses__():
                return logical_type
            elif isinstance(logical_type, str):
                return str_to_logical_type(logical_type)
            else:
                raise TypeError(f"Invalid logical type specified for '{self.series.name}'")
        else:
            return infer_logical_type(self.series)

    def set_semantic_tags(self, semantic_tags):
        """Replace semantic tags with passed values"""
        self._semantic_tags = _parse_semantic_tags(semantic_tags)
        if self.add_standard_tags:
            self._semantic_tags = self._semantic_tags.union(self._logical_type.standard_tags)

    def add_semantic_tags(self, semantic_tags):
        new_tags = _parse_semantic_tags(semantic_tags)
        duplicate_tags = sorted(list(self._semantic_tags.intersection(new_tags)))
        if duplicate_tags:
            warnings.warn(f"Semantic tag(s) '{', '.join(duplicate_tags)}' already present on column '{self.name}'", UserWarning)
        self._semantic_tags = self._semantic_tags.union(new_tags)

    def remove_semantic_tags(self, semantic_tags):
        """Removes specified semantic tags from column and returns a new column"""
        tags_to_remove = _parse_semantic_tags(semantic_tags)
        invalid_tags = sorted(list(tags_to_remove.difference(self._semantic_tags)))
        if invalid_tags:
            raise LookupError(f"Semantic tag(s) '{', '.join(invalid_tags)}' not present on column '{self.name}'")
        standard_tags_to_remove = sorted(list(tags_to_remove.intersection(self._logical_type.standard_tags)))
        if standard_tags_to_remove and self.add_standard_tags:
            warnings.warn(f"Removing standard semantic tag(s) '{', '.join(standard_tags_to_remove)}' from column '{self.name}'",
                          UserWarning)
        new_tags = self._semantic_tags.difference(tags_to_remove)
        return DataColumn(series=self.series,
                          logical_type=self.logical_type,
                          semantic_tags=new_tags,
                          add_standard_tags=False)

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
