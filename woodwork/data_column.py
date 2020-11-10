import warnings

import pandas as pd

from woodwork.config import config
from woodwork.exceptions import (
    ColumnNameMismatchWarning,
    DuplicateTagsWarning,
    StandardTagsRemovalWarning
)
from woodwork.logical_types import (
    Datetime,
    LogicalType,
    NaturalLanguage,
    Ordinal,
    str_to_logical_type
)
from woodwork.utils import (
    _convert_input_to_set,
    _get_ltype_class,
    import_or_none
)

dd = import_or_none('dask.dataframe')


class DataColumn(object):
    def __init__(self, series,
                 logical_type=None,
                 semantic_tags=None,
                 use_standard_tags=True,
                 name=None):
        """Create a DataColumn.

        Args:
            series (pd.Series or dd.Series or pd.api.extensions.ExtensionArray): Series containing the data associated with the column.
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
            name (str, optional): Name of DataColumn. Will overwrite Series name, if it exists.
        """
        self._assigned_name = name
        self._set_series(series)
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

    def __eq__(self, other, deep=False):
        if self.name != other.name:
            return False
        if self.dtype != other.dtype:
            return False
        if self.semantic_tags != other.semantic_tags:
            return False
        if self.logical_type != other.logical_type:
            return False

        # Only check pandas series for equality
        if isinstance(self._series, pd.Series) and isinstance(other.to_series(), pd.Series):
            return self.to_series().equals(other.to_series())
        return True

    def _update_dtype(self):
        """Update the dtype of the underlying series to match the dtype corresponding
        to the LogicalType for the column."""
        if isinstance(self.logical_type, Ordinal):
            self.logical_type._validate_data(self._series)
        if self.logical_type.pandas_dtype != str(self._series.dtype):
            # Update the underlying series
            try:
                if _get_ltype_class(self.logical_type) == Datetime:
                    if dd and isinstance(self._series, dd.Series):
                        name = self._series.name
                        self._series = dd.to_datetime(self._series, format=self.logical_type.datetime_format)
                        self._series.name = name
                    else:
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

    def _set_series(self, series):
        if not (isinstance(series, pd.Series) or
                (dd and isinstance(series, dd.Series)) or
                isinstance(series, pd.api.extensions.ExtensionArray)):
            raise TypeError('Series must be a pandas Series, Dask Series, or a pandas ExtensionArray')

        # pandas ExtensionArrays should be converted to pandas.Series
        if isinstance(series, pd.api.extensions.ExtensionArray):
            series = pd.Series(series, dtype=series.dtype)

        if self._assigned_name is not None and series.name is not None and self._assigned_name != series.name:
            warnings.warn(ColumnNameMismatchWarning().get_warning_message(series.name, self._assigned_name),
                          ColumnNameMismatchWarning)

        series.name = self._assigned_name or series.name
        self._series = series

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
            warnings.warn(DuplicateTagsWarning().get_warning_message(duplicate_tags, self.name),
                          DuplicateTagsWarning)
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
            warnings.warn(StandardTagsRemovalWarning().get_warning_message(standard_tags_to_remove, self.name),
                          StandardTagsRemovalWarning)
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

    def to_series(self):
        """Retrieves the DataColumn's underlying series.

        Note: Do not modify the returned series directly to avoid unexpected behavior

        Returns:
            Series: The underlying series of the DataColumn. Return type will depend on the type
                of series used to create the DataColumn.
        """
        return self._series

    @property
    def shape(self):
        """Returns a tuple representing the dimensionality of the DataTable. If Dask DataFrame, returns
            a Dask `Delayed` object for the number of rows."""
        return self._series.shape

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
        return self._assigned_name or self._series.name

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
    if dd and isinstance(series, dd.Series):
        series = series.get_partition(0).compute()

    for logical_type, func in config.get_option('inference_functions').items():
        if func(series):
            return str_to_logical_type(logical_type)

    return NaturalLanguage
