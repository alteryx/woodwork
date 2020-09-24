import warnings

import pandas.api.types as pdtypes

from woodwork.config import config
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
from woodwork.utils import _convert_input_to_set, col_is_datetime


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
        self._semantic_tags = set()
        self.set_semantic_tags(semantic_tags)

    def __repr__(self):
        msg = u"<DataColumn: {} ".format(self.name)
        msg += u"(Physical Type = {}) ".format(self.dtype)
        msg += u"(Logical Type = {}) ".format(self.logical_type)
        msg += u"(Semantic Tags = {})>".format(self.semantic_tags)
        return msg

    def set_logical_type(self, logical_type, retain_index_tags=True):
        """Update the logical type for the column and return a new column object.

        Args:
            logical_type (LogicalType, str): The new logical type to set for the column.
            retain_index_tags (bool, optional): If True, any 'index' or 'time_index' tags on
                the column will be retained. If False, all tags will be cleared.
                Defaults to True.
        """
        new_logical_type = self._parse_logical_type(logical_type)
        new_col = DataColumn(series=self.series,
                             logical_type=new_logical_type,
                             add_standard_tags=self.add_standard_tags)
        if retain_index_tags and 'index' in self.semantic_tags:
            new_col._set_as_index()
        if retain_index_tags and 'time_index' in self.semantic_tags:
            new_col._set_as_time_index()

        return new_col

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

    def set_semantic_tags(self, semantic_tags, retain_index_tags=True):
        """Replace semantic tags with passed values.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to set for column
            retain_index_tags (bool, optional): If True, any 'index' or 'time_index' tags on
                the column will be retained. If False, all tags will be replaced.
                Defaults to True.
        """
        semantic_tags = _convert_input_to_set(semantic_tags)
        _validate_tags(semantic_tags)
        is_index = 'index' in self._semantic_tags
        is_time_index = 'time_index' in self._semantic_tags
        self._semantic_tags = semantic_tags
        if self.add_standard_tags:
            self._semantic_tags = self._semantic_tags.union(self._logical_type.standard_tags)
        if retain_index_tags and is_index:
            self._set_as_index()
        if retain_index_tags and is_time_index:
            self._set_as_time_index()

    def add_semantic_tags(self, semantic_tags):
        new_tags = _convert_input_to_set(semantic_tags)
        _validate_tags(new_tags)
        duplicate_tags = sorted(list(self._semantic_tags.intersection(new_tags)))
        if duplicate_tags:
            warnings.warn(f"Semantic tag(s) '{', '.join(duplicate_tags)}' already present on column '{self.name}'", UserWarning)
        self._semantic_tags = self._semantic_tags.union(new_tags)

    def reset_semantic_tags(self, retain_index_tags=False):
        """Reset the semantic tags to the default values. The default values
            will be either an empty set or a set of the standard tags based
            on the column logical type, controlled by the add_default_tags
            property.

         Args:
            retain_index_tags (bool, optional): If True, any 'index' or 'time_index' tags on
                the column will be retained. If False, all tags will be cleared.
                Defaults to False.
        """
        new_col = DataColumn(series=self.series,
                             logical_type=self.logical_type,
                             semantic_tags=None,
                             add_standard_tags=self.add_standard_tags)
        if retain_index_tags and 'index' in self.semantic_tags:
            new_col._set_as_index()
        if retain_index_tags and 'time_index' in self.semantic_tags:
            new_col._set_as_time_index()
        return new_col

    def remove_semantic_tags(self, semantic_tags):
        """Removes specified semantic tags from column and returns a new column"""
        tags_to_remove = _convert_input_to_set(semantic_tags)
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

    def _set_as_index(self):
        self._semantic_tags.add('index')

    def _set_as_time_index(self):
        self._semantic_tags.add('time_index')

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


def _validate_tags(semantic_tags):
    """Verify user has not supplied tags that cannot be set directly"""
    if 'index' in semantic_tags:
        raise ValueError("Cannot add 'index' tag directly. To set a column as the index, "
                         "use DataTable.set_index() instead.")
    if 'time_index' in semantic_tags:
        raise ValueError("Cannot add 'time_index' tag directly. To set a column as the time index, "
                         "use DataTable.set_time_index() instead.")


def infer_logical_type(series):
    """Infer logical type for a dataframe column
    Args:
        series (pd.Series): Input Series
    """
    datetime_format = config.get_option('datetime_format')
    natural_language_threshold = config.get_option('natural_language_threshold')

    inferred_type = NaturalLanguage

    if pdtypes.is_string_dtype(series.dtype):
        if col_is_datetime(series, datetime_format):
            inferred_type = Datetime
        else:
            inferred_type = Categorical

            # heuristics to predict this some other than categorical
            sample = series.sample(min(10000, len(series)))

            # catch cases where object dtype cannot be interpreted as a string
            try:
                avg_length = sample.str.len().mean()
                if avg_length > natural_language_threshold:
                    inferred_type = NaturalLanguage
            except AttributeError:
                pass

    elif pdtypes.is_bool_dtype(series.dtype):
        inferred_type = Boolean

    elif pdtypes.is_categorical_dtype(series.dtype):
        inferred_type = Categorical

    elif pdtypes.is_integer_dtype(series.dtype):
        if any(series.dropna() < 0):
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
