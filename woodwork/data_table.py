import warnings

import pandas as pd

from woodwork.config import config
from woodwork.data_column import DataColumn
from woodwork.logical_types import Datetime, LogicalType, str_to_logical_type
from woodwork.utils import _convert_input_to_set, col_is_datetime


class DataTable(object):
    def __init__(self, dataframe,
                 name=None,
                 index=None,
                 time_index=None,
                 semantic_tags=None,
                 logical_types=None,
                 copy_dataframe=False,
                 replace_none=True,
                 add_standard_tags=True):
        """ Create DataTable

        Args:
            dataframe (pd.DataFrame): Dataframe providing the data for the datatable.
            name (str, optional): Name used to identify the datatable.
            index (str, optional): Name of the index column in the dataframe.
            time_index (str, optional): Name of the time index column in the dataframe.
            semantic_tags (dict, optional): Dictionary mapping column names in the dataframe to the
                semantic tags for the column. The keys in the dictionary should be strings
                that correspond to columns in the underlying dataframe.

                There are two options for specifying the dictionary values:
                    (str) If only one semantic tag is being set, a single string can be used as a value.

                    (list[str] or set[str]) If muliple tags are being set, a list or set of strings can be used as the value.

                Semantic tags will be set to an empty set for any column not included in the
                dictionary.
            logical_types (dict[str -> LogicalType], optional): Dictionary mapping column names in
                the dataframe to the LogicalType for the column. LogicalTypes will be inferred
                for any columns not present in the dictionary.
            copy_dataframe (bool, optional): If True, a copy of the input dataframe will be made
                prior to creating the DataTable. Defaults to False, which results in using a
                reference to the input dataframe.
            replace_none (bool, optional): If True, will replace any `None` values in the supplied
                dataframe with `pd.NA`. Defaults to True.
            add_standard_tags (bool, optional): If True, will add standard semantic tags to columns based
                on the inferred or specified logical type for the column. Defaults to True.
        """
        # Check that inputs are valid
        _validate_params(dataframe, name, index, time_index, logical_types, semantic_tags)
        self.add_standard_tags = add_standard_tags

        if copy_dataframe:
            self.dataframe = dataframe.copy()
        else:
            self.dataframe = dataframe

        if replace_none:
            self.dataframe.fillna(pd.NA, inplace=True)

        self.name = name

        # Infer logical types and create columns
        self.columns = self._create_columns(self.dataframe.columns,
                                            logical_types,
                                            semantic_tags,
                                            self.add_standard_tags)
        if index:
            self.set_index(index)
        if time_index:
            self.set_time_index(time_index)

        self._update_dtypes(self.columns)

    def __getitem__(self, key):
        if isinstance(key, list):
            if not all([isinstance(col, str) for col in key]):
                raise KeyError('Column names must be strings')
            invalid_cols = set(key).difference(set(self.columns.keys()))
            if invalid_cols:
                raise KeyError(f"Column(s) '{', '.join(sorted(list(invalid_cols)))}' not found in DataTable")
            return self._new_dt_from_cols(key)
        if not isinstance(key, str):
            raise KeyError('Column name must be a string')
        if key not in self.columns.keys():
            raise KeyError(f"Column with name '{key}' not found in DataTable")
        return self.columns[key]

    @property
    def types(self):
        typing_info = {}
        for dc in self.columns.values():
            typing_info[dc.name] = [dc.dtype, dc.logical_type, dc.semantic_tags]
        df = pd.DataFrame.from_dict(typing_info,
                                    orient='index',
                                    columns=['Physical Type', 'Logical Type', 'Semantic Tag(s)'],
                                    dtype="object")
        df.index.name = 'Data Column'
        return df

    @property
    def ltypes(self):
        return self.types['Logical Type']

    def _create_columns(self,
                        column_names,
                        logical_types,
                        semantic_tags,
                        add_standard_tags):
        """Create a dictionary with column names as keys and new DataColumn objects
            as values, while assigning any values that are passed for logical types or
            semantic tags to the new column."""
        data_columns = {}
        for name in column_names:
            if logical_types and name in logical_types:
                logical_type = logical_types[name]
            else:
                logical_type = None
            if semantic_tags and name in semantic_tags:
                semantic_tag = semantic_tags[name]
            else:
                semantic_tag = None
            dc = DataColumn(self.dataframe[name], logical_type, semantic_tag, add_standard_tags)
            data_columns[dc.name] = dc
        return data_columns

    @property
    def logical_types(self):
        return {dc.name: dc.logical_type for dc in self.columns.values()}

    @property
    def physical_types(self):
        return {dc.name: dc.dtype for dc in self.columns.values()}

    @property
    def semantic_tags(self):
        return {dc.name: dc.semantic_tags for dc in self.columns.values()}

    @property
    def index(self):
        for column in self.columns.values():
            if 'index' in column.semantic_tags:
                return column.name
        return None

    @index.setter
    def index(self, index):
        if self.index and index is None:
            self.remove_semantic_tags({self.index: 'index'})
        elif index is not None:
            self.set_index(index)

    @property
    def time_index(self):
        for column in self.columns.values():
            if 'time_index' in column.semantic_tags:
                return column.name
        return None

    @time_index.setter
    def time_index(self, time_index):
        if self.time_index and time_index is None:
            self.remove_semantic_tags({self.time_index: 'time_index'})
        elif time_index is not None:
            self.set_time_index(time_index)

    def _update_columns(self, new_columns):
        """Update the DataTable columns based on items contained in the
            provided new_columns dictionary"""
        for name, column in new_columns.items():
            self.columns[name] = column

    def set_index(self, index):
        """Set the index column. Adds the 'index' semantic tag to the column and
            clears the tag from any previously set index column"""
        _check_index(self.dataframe, index)
        old_index = self.index
        self.columns[index]._set_as_index()
        if old_index:
            self._update_columns({old_index: self.columns[old_index].remove_semantic_tags('index')})

    def set_time_index(self, time_index):
        """Set the time index column. Adds the 'time_index' semantic tag to the column and
            clears the tag from any previously set index column"""
        _check_time_index(self.dataframe, time_index)
        old_time_index = self.time_index
        self.columns[time_index]._set_as_time_index()
        if old_time_index:
            self._update_columns({old_time_index: self.columns[old_time_index].remove_semantic_tags('time_index')})

    def set_logical_types(self, logical_types, retain_index_tags=True):
        """Update the logical type for any columns names in the provided logical_types
            dictionary. Replaces existing columns with new column objects.

            Args:
                logical_types (dict): A dictionary defining the new logical types for the
                    specified columns.
                retain_index_tags (bool, optional): If True, will retain any index or time_index
                    semantic tags set on the column. If false, will clear all semantic tags. Defaults to
                    True.
        """
        _check_logical_types(self.dataframe, logical_types)
        cols_to_update = {}
        for colname, logical_type in logical_types.items():
            cols_to_update[colname] = self.columns[colname].set_logical_type(logical_type,
                                                                             retain_index_tags=retain_index_tags)
        self._update_columns(cols_to_update)
        self._update_dtypes(cols_to_update)

    def _update_dtypes(self, cols_to_update):
        """Update the dtypes of the underlying dataframe to match the dtypes corresponding
            to the LogicalType for the column."""
        for name, column in cols_to_update.items():
            if column.logical_type.pandas_dtype != str(self.dataframe[name].dtype):
                # Update the underlying dataframe
                try:
                    if column.logical_type == Datetime:
                        self.dataframe[name] = pd.to_datetime(self.dataframe[name], format=config.get_option('datetime_format'))
                    else:
                        self.dataframe[name] = self.dataframe[name].astype(column.logical_type.pandas_dtype)
                except TypeError:
                    error_msg = f'Error converting datatype for column {name} from type {str(self.dataframe[name].dtype)} ' \
                        f'to type {column.logical_type.pandas_dtype}. Please confirm the underlying data is consistent with ' \
                        f'logical type {column.logical_type}.'
                    raise TypeError(error_msg)
                # Update the column object since .astype returns a new series object
                column.series = self.dataframe[name]

    def add_semantic_tags(self, semantic_tags):
        """Adds specified semantic tags to columns. Will retain any previously set values."""
        _check_semantic_tags(self.dataframe, semantic_tags)
        for name in semantic_tags.keys():
            self.columns[name].add_semantic_tags(semantic_tags[name])

    def remove_semantic_tags(self, semantic_tags):
        """Remove the semantic tags for any column names in the provided semantic_tags
            dictionary. Replaces the column with a new column object."""
        _check_semantic_tags(self.dataframe, semantic_tags)
        cols_to_update = {}
        for colname, tags in semantic_tags.items():
            cols_to_update[colname] = self.columns[colname].remove_semantic_tags(tags)
        self._update_columns(cols_to_update)

    def set_semantic_tags(self, semantic_tags, retain_index_tags=True):
        """Update the semantic tags for any column names in the provided semantic_tags
            dictionary. Replaces the existing semantic tags with the new values.
        Args:
            semantic_tags (dict): A dictionary defining the new semantic_tags for the
                specified columns.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will replace all semantic tags. Defaults to
                True.
        """
        _check_semantic_tags(self.dataframe, semantic_tags)
        for name in semantic_tags.keys():
            self.columns[name].set_semantic_tags(semantic_tags[name], retain_index_tags)

    def reset_semantic_tags(self, columns=None, retain_index_tags=False):
        """Reset the semantic tags for the specified columns to the default values.
            The default values will be either an empty set or a set of the standard
            tags based on the column logical type, controlled by the add_default_tags
            property on the table. Columns names can be provided as a single string,
            a list of strings or a set of strings. If columns is not specified,
            tags will be reset for all columns.
        Args:
            columns (str/list/set): The columns for which the semantic tags should be reset.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will clear all semantic tags. Defaults to
                False.
        """
        columns = _convert_input_to_set(columns, "columns")
        cols_not_found = sorted(list(columns.difference(set(self.dataframe.columns))))
        if cols_not_found:
            raise LookupError("Input contains columns that are not present in "
                              f"dataframe: '{', '.join(cols_not_found)}'")
        if not columns:
            columns = self.columns.keys()
        cols_to_update = {}
        for colname in columns:
            cols_to_update[colname] = self.columns[colname].reset_semantic_tags(retain_index_tags)
        self._update_columns(cols_to_update)

    @property
    def df(self):
        return self.dataframe

    def to_pandas_dataframe(self):
        return self.dataframe

    def select(self, include):
        """Create a DataTable including only columns whose logical type and
            semantic tags are specified in the list of types and tags to include.

        Args:
            include  (str or LogicalType or list[str or LogicalType]):
                Logical types and semantic tags to include in the DataTable.

        Returns:
            DataTable:
                The subset of the original DataTable that contains just the
                logical types and semantic tags in `include`.
        """
        if not isinstance(include, list):
            include = [include]

        ltypes_used = set()
        ltypes_in_dt = {col.logical_type for col in self.columns.values()}

        tags_used = set()
        tags_in_dt = {tag for col in self.columns.values() for tag in col.semantic_tags}

        unused_selectors = []

        for selector in include:
            if selector in LogicalType.__subclasses__():
                if selector in ltypes_in_dt:
                    ltypes_used.add(selector)
                else:
                    unused_selectors.append(str(selector))
            elif isinstance(selector, str):
                # If the str is a viable ltype, it'll take precedence
                # but if it's not present, we'll check if it's a tag
                ltype = str_to_logical_type(selector, raise_error=False)
                if ltype and ltype in ltypes_in_dt:
                    ltypes_used.add(ltype)
                    continue
                elif selector in tags_in_dt:
                    tags_used.add(selector)
                else:
                    unused_selectors.append(selector)
            else:
                raise TypeError(f"Invalid selector used in include: {selector} must be either a string or LogicalType")

        if unused_selectors:
            not_present_str = ', '.join(sorted(unused_selectors))
            warnings.warn(f'The following selectors were not present in your DataTable: {not_present_str}')

        cols_to_include = []
        for col_name, col in self.columns.items():
            if col.logical_type in ltypes_used or col.semantic_tags.intersection(tags_used):
                cols_to_include.append(col_name)

        return self._new_dt_from_cols(cols_to_include)

    def select_ltypes(self, include):
        """Create a DataTable that includes only columns whose logical types are specified here.
            Will not include any column, including indices, whose logical type is not specified.

        Args:
            include (str or LogicalType or list[str or LogicalType]):
                Logical types to include in the DataTable

        Returns:
            DataTable:
                The subset of the original DataTable that contains just the ltypes in `include`.
        """
        if not isinstance(include, list):
            include = [include]

        ltypes_to_include = set()
        for ltype in include:
            if ltype in LogicalType.__subclasses__():
                ltypes_to_include.add(ltype)
            elif isinstance(ltype, str):
                ltypes_to_include.add(str_to_logical_type(ltype))
            else:
                raise TypeError(f"Invalid logical type specified: {ltype}")

        ltypes_present = {col.logical_type for col in self.columns.values()}
        unused_ltypes = ltypes_to_include - ltypes_present

        if unused_ltypes:
            not_present_str = ', '.join(sorted([str(ltype) for ltype in unused_ltypes]))
            warnings.warn(f'The following logical types were not present in your DataTable: {not_present_str}')

        cols_to_include = [col_name for col_name, col in self.columns.items()
                           if col.logical_type in ltypes_to_include]

        return self._new_dt_from_cols(cols_to_include)

    def select_semantic_tags(self, include):
        """Create a DataTable that includes only columns that have at least one of the semantic tags
            specified here. The new DataTable with retain any logical types or semantic tags from
            the original DataTable.

            Args:
                include (str or list[str] or set[str]): Semantic tags to include in the DataTable

            Returns:
                DataTable:
                    The subset of the original DataTable that contains just the semantic tags in `include`.
        """
        include = _convert_input_to_set(include, 'include parameter')

        include = set(include)
        cols_to_include = []
        for col_name, tags in self.semantic_tags.items():
            intersection = tags.intersection(include)
            if intersection:
                cols_to_include.append(col_name)

        new_dt = self._new_dt_from_cols(cols_to_include)

        tags_present = {tag for col in new_dt.columns.values() for tag in col.semantic_tags}
        unused_tags = include - tags_present

        if unused_tags:
            not_present_str = ', '.join(sorted(list(unused_tags)))
            warnings.warn(f'The following semantic tags were not present in your DataTable: {not_present_str}')

        return new_dt

    def _new_dt_from_cols(self, cols_to_include):
        """Creates a new DataTable from a list of column names, retaining all types, indices, and name of original DataTable"""
        assert all([col_name in self.columns for col_name in cols_to_include])

        new_semantic_tags = {col_name: semantic_tag_set for col_name, semantic_tag_set
                             in self.semantic_tags.items() if col_name in cols_to_include}
        new_logical_types = {col_name: logical_type for col_name, logical_type
                             in self.logical_types.items() if col_name in cols_to_include}
        new_index = self.index if self.index in cols_to_include else None
        new_time_index = self.time_index if self.time_index in cols_to_include else None
        # Remove 'index' or 'time_index' from semantic tags, if present as those can't be set directly during init
        if new_index:
            new_semantic_tags[new_index] = new_semantic_tags[new_index].difference({'index'})
        if new_time_index:
            new_semantic_tags[new_time_index] = new_semantic_tags[new_time_index].difference({'time_index'})

        return DataTable(self.dataframe[cols_to_include],
                         name=self.name,
                         index=new_index,
                         time_index=new_time_index,
                         semantic_tags=new_semantic_tags,
                         logical_types=new_logical_types,
                         copy_dataframe=True)


def _validate_params(dataframe, name, index, time_index, logical_types, semantic_tags):
    """Check that values supplied during DataTable initialization are valid"""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError('Dataframe must be a pandas.DataFrame')
    _check_unique_column_names(dataframe)
    if name and not isinstance(name, str):
        raise TypeError('DataTable name must be a string')
    if index:
        _check_index(dataframe, index)
    if time_index:
        _check_time_index(dataframe, time_index)
    if logical_types:
        _check_logical_types(dataframe, logical_types)
    if semantic_tags:
        _check_semantic_tags(dataframe, semantic_tags)


def _check_unique_column_names(dataframe):
    if not dataframe.columns.is_unique:
        raise IndexError('Dataframe cannot contain duplicate columns names')


def _check_index(dataframe, index):
    if not isinstance(index, str):
        raise TypeError('Index column name must be a string')
    if index not in dataframe.columns:
        raise LookupError(f'Specified index column `{index}` not found in dataframe')
    if not dataframe[index].is_unique:
        raise IndexError('Index column must be unique')


def _check_time_index(dataframe, time_index):
    if not isinstance(time_index, str):
        raise TypeError('Time index column name must be a string')
    if time_index not in dataframe.columns:
        raise LookupError(f'Specified time index column `{time_index}` not found in dataframe')
    if not col_is_datetime(dataframe[time_index]):
        raise TypeError('Time index column must contain datetime values')


def _check_logical_types(dataframe, logical_types):
    if not isinstance(logical_types, dict):
        raise TypeError('logical_types must be a dictionary')
    cols_not_found = set(logical_types.keys()).difference(set(dataframe.columns))
    if cols_not_found:
        raise LookupError('logical_types contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')


def _check_semantic_tags(dataframe, semantic_tags):
    if not isinstance(semantic_tags, dict):
        raise TypeError('semantic_tags must be a dictionary')
    cols_not_found = set(semantic_tags.keys()).difference(set(dataframe.columns))
    if cols_not_found:
        raise LookupError('semantic_tags contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')
