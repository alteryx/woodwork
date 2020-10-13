import warnings

import pandas as pd
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
    Ordinal,
    Timedelta,
    WholeNumber,
    str_to_logical_type
)
from woodwork.utils import (
    _convert_input_to_set,
    _get_ltype_class,
    col_is_datetime
)


class DataColumn(object):
    def __init__(self, series,
                 logical_type=None,
                 semantic_tags=None,
                 use_standard_tags=True):
        """Create a DataColumn.

        Args:
            series (pd.Series): Series containing the data associated with the column.
            logical_type (LogicalType, optional): The logical type that should be assigned
                to the column. If no value is provided, the LogicalType for the series will
                be inferred.
            semantic_tags (str or list or set, optional): Semantic tags to assign to the column.
                Defaults to an empty set if not specified. There are two options for
                specifying the semantic tags:
                (str) If only one semantic tag is being set, a single string can be passed.
                (list or set) If multiple tags are being set, a list or set of strings can be passed.
            use_standard_tags (bool, optional): If True, will add standard semantic tags to columns based
                on the inferred or specified logical type for the column. Defaults to True.
        """
        self._series = series
        self.use_standard_tags = use_standard_tags
        self._logical_type = self._parse_logical_type(logical_type)
        semantic_tags = _convert_input_to_set(semantic_tags)
        _validate_tags(semantic_tags)
        if use_standard_tags:
            semantic_tags = semantic_tags.union(self.logical_type.standard_tags)
        self._semantic_tags = semantic_tags
        self._update_dtype()

    def __repr__(self):
        msg = u"<DataColumn: {} ".format(self.name)
        msg += u"(Physical Type = {}) ".format(self.dtype)
        msg += u"(Logical Type = {}) ".format(self.logical_type)
        msg += u"(Semantic Tags = {})>".format(self.semantic_tags)
        return msg

    def _update_dtype(self):
        """Update the dtype of the underlying series to match the dtype corresponding
        to the LogicalType for the column."""
        if isinstance(self.logical_type, Ordinal):
            self.logical_type._validate_data(self._series)
        if self.logical_type.pandas_dtype != str(self._series.dtype):
            # Update the underlying series
            try:
                if _get_ltype_class(self.logical_type) == Datetime:
                    self._series = pd.to_datetime(self._series, format=self.logical_type.datetime_format)
                else:
                    self._series = self._series.astype(self.logical_type.pandas_dtype)
            except TypeError:
                error_msg = f'Error converting datatype for column {self.name} from type {str(self._series.dtype)} ' \
                    f'to type {self.logical_type.pandas_dtype}. Please confirm the underlying data is consistent with ' \
                    f'logical type {self.logical_type}.'
                raise TypeError(error_msg)

    def set_logical_type(self, logical_type, retain_index_tags=True):
        """Update the logical type for the column and return a new DataColumn object.

        Args:
            logical_type (LogicalType, str): The new logical type to set for the column.
            retain_index_tags (bool, optional): If True, any 'index' or 'time_index' tags on
                the column will be retained. If False, all tags will be cleared.
                Defaults to True.

        Returns:
            woodwork.DataColumn: DataColumn with updated logical type.
        """
        new_logical_type = self._parse_logical_type(logical_type)
        new_col = DataColumn(series=self._series,
                             logical_type=new_logical_type,
                             use_standard_tags=self.use_standard_tags)
        if retain_index_tags and 'index' in self.semantic_tags:
            new_col._set_as_index()
        if retain_index_tags and 'time_index' in self.semantic_tags:
            new_col._set_as_time_index()

        return new_col

    def _parse_logical_type(self, logical_type):
        if logical_type:
            if isinstance(logical_type, str):
                logical_type = str_to_logical_type(logical_type)
            ltype_class = _get_ltype_class(logical_type)
            if ltype_class == Ordinal and not isinstance(logical_type, Ordinal):
                raise TypeError("Must use an Ordinal instance with order values defined")
            if ltype_class in LogicalType.__subclasses__():
                return logical_type
            else:
                raise TypeError(f"Invalid logical type specified for '{self.name}'")
        else:
            return infer_logical_type(self._series)

    def set_semantic_tags(self, semantic_tags, retain_index_tags=True):
        """Replace current semantic tags with new values and return a new DataColumn object.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to set for column
            retain_index_tags (bool, optional): If True, any 'index' or 'time_index' tags on
                the column will be retained. If False, all tags will be replaced.
                Defaults to True.

        Returns:
            woodwork.DataColumn: DataColumn with specified semantic tags.
        """
        semantic_tags = _convert_input_to_set(semantic_tags)
        _validate_tags(semantic_tags)
        is_index = 'index' in self._semantic_tags
        is_time_index = 'time_index' in self._semantic_tags
        new_col = DataColumn(series=self._series,
                             logical_type=self.logical_type,
                             semantic_tags=semantic_tags,
                             use_standard_tags=self.use_standard_tags)
        if new_col.use_standard_tags:
            new_col._semantic_tags = new_col._semantic_tags.union(new_col._logical_type.standard_tags)
        if retain_index_tags and is_index:
            new_col._set_as_index()
        if retain_index_tags and is_time_index:
            new_col._set_as_time_index()
        return new_col

    def add_semantic_tags(self, semantic_tags):
        """Add the specified semantic tags to the column and return a new DataColumn object.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to add to the column

        Returns:
            woodwork.DataColumn: DataColumn with specified semantic tags added.
        """
        new_tags = _convert_input_to_set(semantic_tags)
        _validate_tags(new_tags)
        duplicate_tags = sorted(list(self._semantic_tags.intersection(new_tags)))
        if duplicate_tags:
            warnings.warn(f"Semantic tag(s) '{', '.join(duplicate_tags)}' already present on column '{self.name}'", UserWarning)
        new_col_tags = self._semantic_tags.union(new_tags)
        new_col = DataColumn(series=self._series,
                             logical_type=self.logical_type,
                             semantic_tags=new_col_tags,
                             use_standard_tags=self.use_standard_tags)
        return new_col

    def reset_semantic_tags(self, retain_index_tags=False):
        """Reset the semantic tags to the default values. The default values
        will be either an empty set or a set of the standard tags based on the
        column logical type, controlled by the use_standard_tags property.

        Args:
            retain_index_tags (bool, optional): If True, any 'index' or
                'time_index' tags on the column will be retained. If False,
                all tags will be cleared. Defaults to False.

        Returns:
            woodwork.DataColumn: DataColumn with reset semantic tags.
        """
        new_col = DataColumn(series=self._series,
                             logical_type=self.logical_type,
                             semantic_tags=None,
                             use_standard_tags=self.use_standard_tags)
        if retain_index_tags and 'index' in self.semantic_tags:
            new_col._set_as_index()
        if retain_index_tags and 'time_index' in self.semantic_tags:
            new_col._set_as_time_index()
        return new_col

    def remove_semantic_tags(self, semantic_tags):
        """Removes specified semantic tags from column and returns a new column.

        Args:
            semantic_tags (str/list/set): Semantic tag(s) to remove from the column.

        Returns:
            woodwork.DataColumn: DataColumn with specified tags removed.
        """
        tags_to_remove = _convert_input_to_set(semantic_tags)
        invalid_tags = sorted(list(tags_to_remove.difference(self._semantic_tags)))
        if invalid_tags:
            raise LookupError(f"Semantic tag(s) '{', '.join(invalid_tags)}' not present on column '{self.name}'")
        standard_tags_to_remove = sorted(list(tags_to_remove.intersection(self._logical_type.standard_tags)))
        if standard_tags_to_remove and self.use_standard_tags:
            warnings.warn(f"Removing standard semantic tag(s) '{', '.join(standard_tags_to_remove)}' from column '{self.name}'",
                          UserWarning)
        new_tags = self._semantic_tags.difference(tags_to_remove)
        return DataColumn(series=self._series,
                          logical_type=self.logical_type,
                          semantic_tags=new_tags,
                          use_standard_tags=False)

    def _set_as_index(self):
        self._semantic_tags = self._semantic_tags.difference(self._logical_type.standard_tags)
        self._semantic_tags.add('index')

    def _set_as_time_index(self):
        self._semantic_tags.add('time_index')

    def _is_numeric(self):
        return 'numeric' in self.logical_type.standard_tags

    def _is_categorical(self):
        return 'category' in self.logical_type.standard_tags

    def to_pandas(self, copy=False):
        """Retrieves the DataColumn's underlying series.

        Note: Do not modify the series unless copy=True has been set to avoid unexpected behavior

        Args:
            copy (bool): If set to True, returns a copy of the underlying series.
                If False, will return a reference to the DataColumn's series, which,
                if modified, can cause unexpected behavior in the DataColumn.
                Defaults to False.

        Returns:
            pandas.Series: The underlying series of the DataColumn
        """
        if copy:
            return self._series.copy()
        return self._series

    @property
    def logical_type(self):
        """The logical type for the column"""
        return self._logical_type

    @property
    def semantic_tags(self):
        """The set of semantic tags currently assigned to the column"""
        return self._semantic_tags

    @property
    def name(self):
        """The name of the column"""
        return self._series.name

    @property
    def dtype(self):
        """The dtype of the underlying series"""
        return self._series.dtype


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
    natural_language_threshold = config.get_option('natural_language_threshold')

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
