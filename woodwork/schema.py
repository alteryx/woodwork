import collections
import copy

import pandas as pd

import woodwork as ww
from woodwork.exceptions import ColumnNotPresentError
from woodwork.schema_column import (
    _add_semantic_tags,
    _get_column_dict,
    _remove_semantic_tags,
    _reset_semantic_tags,
    _set_semantic_tags
)
from woodwork.type_sys.utils import _get_ltype_class
from woodwork.utils import _convert_input_to_set


class Schema(object):
    def __init__(self, column_names,
                 logical_types,
                 name=None,
                 index=None,
                 time_index=None,
                 semantic_tags=None,
                 table_metadata=None,
                 column_metadata=None,
                 use_standard_tags=True,
                 column_descriptions=None):
        """Create Schema

        Args:
            column_names (list, set): The columns present in the Schema.
            logical_types (dict[str -> LogicalType]): Dictionary mapping column names in
                the Schema to the LogicalType for the column. All columns present in the
                Schema must be present in the logical_types dictionary.
            name (str, optional): Name used to identify the Schema.
            index (str, optional): Name of the index column.
            time_index (str, optional): Name of the time index column.
            semantic_tags (dict, optional): Dictionary mapping column names in the Schema to the
                semantic tags for the column. The keys in the dictionary should be strings
                that correspond to columns in the Schema. There are two options for
                specifying the dictionary values:
                (str): If only one semantic tag is being set, a single string can be used as a value.
                (list[str] or set[str]): If multiple tags are being set, a list or set of strings can be
                used as the value.
                Semantic tags will be set to an empty set for any column not included in the
                dictionary.
            table_metadata (dict[str -> json serializable], optional): Dictionary containing extra metadata for the Schema.
            column_metadata (dict[str -> dict[str -> json serializable]], optional): Dictionary mapping column names
                to that column's metadata dictionary.
            use_standard_tags (bool, optional): If True, will add standard semantic tags to columns based
                specified logical type for the column. Defaults to True.
            column_descriptions (dict[str -> str], optional): Dictionary mapping column names to column descriptions.
        """
        # Check that inputs are valid
        _validate_params(column_names, name, index, time_index, logical_types,
                         table_metadata, column_metadata, semantic_tags, column_descriptions)

        self.name = name
        self.use_standard_tags = use_standard_tags

        # Infer logical types and create columns
        self.columns = self._create_columns(column_names,
                                            logical_types,
                                            semantic_tags,
                                            use_standard_tags,
                                            column_descriptions,
                                            column_metadata)
        if index is not None:
            self.set_index(index)

        if time_index is not None:
            self.set_time_index(time_index)

        self.metadata = table_metadata or {}

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.index != other.index:
            return False
        if self.time_index != other.time_index:
            return False
        if self.use_standard_tags != other.use_standard_tags:
            return False
        if self.columns != other.columns:
            return False
        if self.metadata != other.metadata:
            return False

        return True

    def __repr__(self):
        """A string representation of a Schema containing typing information."""
        return repr(self._get_typing_info())

    def _repr_html_(self):
        """An HTML representation of a Schema for IPython.display in Jupyter Notebooks
        containing typing information and a preview of the data."""
        return self._get_typing_info().to_html()

    @property
    def types(self):
        """DataFrame containing the physical dtypes, logical types and semantic
        tags for the Schema."""
        return self._get_typing_info()

    def _get_typing_info(self):
        """Creates a DataFrame that contains the typing information for a Schema."""
        typing_info = {}
        for col_name, col_dict in self.columns.items():

            types = [col_dict['logical_type'], str(list(col_dict['semantic_tags']))]
            typing_info[col_name] = types

        columns = ['Logical Type', 'Semantic Tag(s)']

        df = pd.DataFrame.from_dict(typing_info,
                                    orient='index',
                                    columns=columns,
                                    dtype="object")
        df.index.name = 'Column'
        return df

    @property
    def logical_types(self):
        """A dictionary containing logical types for each column"""
        return {col_name: col['logical_type'] for col_name, col in self.columns.items()}

    @property
    def semantic_tags(self):
        """A dictionary containing semantic tags for each column"""
        return {col_name: col['semantic_tags'] for col_name, col in self.columns.items()}

    @property
    def index(self):
        """The index column for the table"""
        for col_name, column in self.columns.items():
            if 'index' in column['semantic_tags']:
                return col_name
        return None

    @property
    def time_index(self):
        """The time index column for the table"""
        for col_name, column in self.columns.items():
            if 'time_index' in column['semantic_tags']:
                return col_name
        return None

    def set_types(self, logical_types=None, semantic_tags=None, retain_index_tags=True):
        """Update the logical type and semantic tags for any columns names in the provided types dictionaries,
        updating the Schema at those columns.

        Args:
            logical_types (dict[str -> LogicalType], optional): A dictionary defining the new logical types for the
                specified columns.
            semantic_tags (dict[str -> str/list/set], optional): A dictionary defining the new semantic_tags for the
                specified columns.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will replace all semantic tags any time a column's
                semantic tags or logical type changes. Defaults to True.
        """
        logical_types = logical_types or {}
        _check_logical_types(self.columns.keys(), logical_types, require_all_cols=False)

        semantic_tags = semantic_tags or {}
        _check_semantic_tags(self.columns.keys(), semantic_tags)

        for col_name in (logical_types.keys() | semantic_tags.keys()):
            original_tags = self.semantic_tags[col_name]

            # Update Logical Type for the Schema, getting new semantic tags
            new_logical_type = logical_types.get(col_name)
            if new_logical_type is not None:
                self.columns[col_name]['logical_type'] = new_logical_type

            # Get new semantic tags, removing existing tags
            new_semantic_tags = semantic_tags.get(col_name)
            if new_semantic_tags is None:
                new_semantic_tags = _reset_semantic_tags(new_logical_type.standard_tags, self.use_standard_tags)
            else:
                new_semantic_tags = _set_semantic_tags(new_semantic_tags,
                                                       self.logical_types[col_name].standard_tags,
                                                       self.use_standard_tags)
                _validate_not_setting_index_tags(new_semantic_tags, col_name)

            # Update the tags for the Schema
            self.columns[col_name]['semantic_tags'] = new_semantic_tags

            if retain_index_tags and 'index' in original_tags:
                self._set_index_tags(col_name)
            if retain_index_tags and 'time_index' in original_tags:
                self._set_time_index_tags(col_name)

    def add_semantic_tags(self, semantic_tags):
        """Adds specified semantic tags to columns, updating the Woodwork typing information.
        Will retain any previously set values.

        Args:
            semantic_tags (dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataFrame to the tags that should be added to the column's semantic tags
        """
        _check_semantic_tags(self.columns.keys(), semantic_tags)
        for col_name, tags_to_add in semantic_tags.items():
            tags_to_add = _convert_input_to_set(tags_to_add)
            _validate_not_setting_index_tags(tags_to_add, col_name)
            new_semantic_tags = _add_semantic_tags(tags_to_add, self.semantic_tags[col_name], col_name)
            self.columns[col_name]['semantic_tags'] = new_semantic_tags

    def remove_semantic_tags(self, semantic_tags):
        """Remove the semantic tags for any column names in the provided semantic_tags
        dictionary, updating the Woodwork typing information. Including `index` or `time_index`
        tags will set the Woodwork index or time index to None for the DataFrame.

        Args:
            semantic_tags (dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataFrame to the tags that should be removed from the column's semantic tags
        """
        _check_semantic_tags(self.columns.keys(), semantic_tags)
        for col_name, tags_to_remove in semantic_tags.items():
            standard_tags = self.logical_types[col_name].standard_tags
            tags_to_remove = _convert_input_to_set(tags_to_remove)

            new_semantic_tags = _remove_semantic_tags(tags_to_remove,
                                                      self.semantic_tags[col_name],
                                                      col_name,
                                                      standard_tags,
                                                      self.use_standard_tags)
            # If the index is removed, reinsert any standard tags not explicitly removed
            original_tags = self.semantic_tags[col_name]
            if self.use_standard_tags and 'index' in original_tags and 'index' not in new_semantic_tags:
                standard_tags_removed = tags_to_remove.intersection(standard_tags)
                standard_tags_to_reinsert = standard_tags.difference(standard_tags_removed)
                new_semantic_tags = new_semantic_tags.union(standard_tags_to_reinsert)
            self.columns[col_name]['semantic_tags'] = new_semantic_tags

    def reset_semantic_tags(self, columns=None, retain_index_tags=False):
        """Reset the semantic tags for the specified columns to the default values.
        The default values will be either an empty set or a set of the standard tags
        based on the column logical type, controlled by the use_standard_tags property on the table.
        Column names can be provided as a single string, a list of strings or a set of strings.
        If columns is not specified, tags will be reset for all columns.

        Args:
            columns (str/list/set, optional): The columns for which the semantic tags should be reset.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will clear all semantic tags. Defaults to
                False.
        """
        columns = _convert_input_to_set(columns, "columns")
        cols_not_found = sorted(list(columns.difference(set(self.columns.keys()))))
        if cols_not_found:
            raise LookupError("Input contains columns that are not present in "
                              f"dataframe: '{', '.join(cols_not_found)}'")
        if not columns:
            columns = self.columns.keys()

        for col_name in columns:
            original_tags = self.semantic_tags[col_name]
            new_semantic_tags = _reset_semantic_tags(self.logical_types[col_name].standard_tags, self.use_standard_tags)
            self.columns[col_name]['semantic_tags'] = new_semantic_tags

            if retain_index_tags and 'index' in original_tags:
                self._set_index_tags(col_name)
            if retain_index_tags and 'time_index' in original_tags:
                self._set_time_index_tags(col_name)

    def _create_columns(self,
                        column_names,
                        logical_types,
                        semantic_tags,
                        use_standard_tags,
                        column_descriptions,
                        column_metadata):
        """Create a dictionary with column names as keys and new column dictionaries holding
        each column's typing information as values."""
        columns = {}
        for name in column_names:
            semantic_tags_for_col = _convert_input_to_set((semantic_tags or {}).get(name),
                                                          error_language=f'semantic_tags for {name}')
            _validate_not_setting_index_tags(semantic_tags_for_col, name)
            description = (column_descriptions or {}).get(name)
            metadata_for_col = (column_metadata or {}).get(name)

            columns[name] = _get_column_dict(name,
                                             logical_types.get(name),
                                             semantic_tags=semantic_tags_for_col,
                                             use_standard_tags=use_standard_tags,
                                             description=description,
                                             metadata=metadata_for_col)
        return columns

    def set_index(self, new_index):
        """Sets the index. Handles setting a new index, updating the index, or removing the index.

        Args:
            new_index (str): Name of the new index column. Must be present in the Schema.
                If None, will remove the index.
        """
        old_index = self.index
        if old_index is not None:
            self.remove_semantic_tags({old_index: 'index'})
        if new_index is not None:
            _check_index(self.columns.keys(), new_index)
            self._set_index_tags(new_index)

    def set_time_index(self, new_time_index):
        """Set the time index. Adds the 'time_index' semantic tag to the column and
        clears the tag from any previously set index column

        Args:
            new_time_index (str): The name of the column to set as the time index.
                If None, will remove the time_index.
        """
        old_time_index = self.time_index
        if old_time_index is not None:
            self.remove_semantic_tags({old_time_index: 'time_index'})
        if new_time_index is not None:
            _check_time_index(self.columns.keys(), new_time_index, self.logical_types.get(new_time_index))
            self._set_time_index_tags(new_time_index)

    def rename(self, columns):
        """Renames columns in a Schema

        Args:
            columns (dict[str -> str]): A dictionary mapping current column names to new column names.

        Returns:
            woodwork.Schema: Schema with the specified columns renamed.
        """
        if not isinstance(columns, dict):
            raise TypeError('columns must be a dictionary')

        for old_name, new_name in columns.items():
            if old_name not in self.columns:
                raise ColumnNotPresentError(f"Column to rename must be present. {old_name} cannot be found.")
            if new_name in self.columns and new_name not in columns.keys():
                raise ValueError(f"The column {new_name} is already present. Please choose another name to rename {old_name} to or also rename {old_name}.")

        if len(columns) != len(set(columns.values())):
            raise ValueError('New columns names must be unique from one another.')

        new_schema = self._get_subset_schema(list(self.columns.keys()))

        cols_to_update = {}
        for old_name, new_name in columns.items():
            col = new_schema.columns.pop(old_name)
            cols_to_update[new_name] = col

        new_schema.columns.update(cols_to_update)
        return new_schema

    def _set_index_tags(self, index):
        """Updates the semantic tags of the index by removing any standard tags
        before adding the 'index' tag."""
        column_dict = self.columns[index]

        standard_tags = column_dict['logical_type'].standard_tags
        new_tags = column_dict['semantic_tags'].difference(standard_tags)
        new_tags.add('index')

        self.columns[index]['semantic_tags'] = new_tags

    def _set_time_index_tags(self, time_index):
        self.columns[time_index]['semantic_tags'].add('time_index')

    def _filter_cols(self, include, col_names=False):
        """Return list of columns filtered with any of: semantic tags, LogicalTypes, column names

        Args:
            include (str or LogicalType or list[str or LogicalType]): parameter or list of parameters to
                filter columns by. Can be Logical Types or Semantic Tags.

            col_names (bool): Specifies whether to filter columns by name. Defaults to False.

        Returns:
            List[str] of column names that fit into filter.
        """
        if not isinstance(include, list):
            include = [include]

        ltypes_used = set()
        ltypes_in_schema = {_get_ltype_class(col['logical_type']) for col in self.columns.values()}

        tags_used = set()
        tags_in_schema = {tag for col in self.columns.values() for tag in col['semantic_tags']}

        cols_to_include = set()

        for selector in include:
            # Determine if the selector is a registered, uninstantiated LogicalType
            maybe_ltype = selector
            if isinstance(selector, str):
                # Convert possible string to LogicalType - unregistered LogicalTypes return None
                maybe_ltype = ww.type_system.str_to_logical_type(selector, raise_error=False)
            # Get the class - unregistered LogicalTypes return LogicalTypeMetaClass
            maybe_ltype_class = _get_ltype_class(maybe_ltype)

            if maybe_ltype_class in ww.type_system.registered_types:
                if maybe_ltype not in ww.type_system.registered_types:
                    raise TypeError(f"Invalid selector used in include: {maybe_ltype} cannot be instantiated")
                if maybe_ltype in ltypes_in_schema:
                    ltypes_used.add(maybe_ltype)
            elif maybe_ltype_class == ww.logical_types.LogicalType.__class__:
                raise TypeError(f"Specified LogicalType selector {maybe_ltype} is not registered in Woodwork's type system.")

            # Hashability as a proxy for whether a selector is possibly a semantic tag or column name
            if not isinstance(selector, collections.Hashable):
                raise TypeError(f"Invalid selector used in include: {selector} must be a "
                                "string, uninstantiated and registered LogicalType, or valid column name")
            # Determine if the selector is a semantic tag
            if selector in tags_in_schema:
                tags_used.add(selector)
            # Determine if the selector is a column name
            if col_names and selector in self.columns:
                cols_to_include.add(selector)

        for col_name, col in self.columns.items():
            if _get_ltype_class(col['logical_type']) in ltypes_used or col['semantic_tags'].intersection(tags_used):
                cols_to_include.add(col_name)

        return list(cols_to_include)

    def _get_subset_schema(self, subset_cols):
        """Creates a new Schema with specified columns, retaining typing information.

        Args:
            subset_cols (list[str]): subset of columns from which to create the new Schema
        Returns:
            Schema: New Schema with attributes from original Schema
        """
        new_logical_types = {}
        new_semantic_tags = {}
        new_column_descriptions = {}
        new_column_metadata = {}
        for col_name in subset_cols:
            col = col = self.columns[col_name]

            new_logical_types[col_name] = col['logical_type']
            new_semantic_tags[col_name] = col['semantic_tags']
            new_column_descriptions[col_name] = col['description']
            new_column_metadata[col_name] = col['metadata']

        new_index = self.index if self.index in subset_cols else None
        new_time_index = self.time_index if self.time_index in subset_cols else None
        if new_index is not None:
            new_semantic_tags[new_index] = new_semantic_tags[new_index].difference({'index'})
        if new_time_index is not None:
            new_semantic_tags[new_time_index] = new_semantic_tags[new_time_index].difference({'time_index'})

        return Schema(subset_cols,
                      new_logical_types,
                      name=self.name,
                      index=new_index,
                      time_index=new_time_index,
                      semantic_tags=copy.deepcopy(new_semantic_tags),
                      use_standard_tags=self.use_standard_tags,
                      table_metadata=copy.deepcopy(self.metadata),
                      column_metadata=copy.deepcopy(new_column_metadata),
                      column_descriptions=new_column_descriptions)


def _validate_params(column_names, name, index, time_index, logical_types,
                     table_metadata, column_metadata, semantic_tags, column_descriptions):
    """Check that values supplied during Schema initialization are valid"""
    _check_column_names(column_names)
    if name and not isinstance(name, str):
        raise TypeError('Schema name must be a string')
    if index is not None:
        _check_index(column_names, index)
    if logical_types:
        _check_logical_types(column_names, logical_types)
    if table_metadata:
        _check_table_metadata(table_metadata)
    if column_metadata:
        _check_column_metadata(column_names, column_metadata)
    if time_index is not None:
        _check_time_index(column_names, time_index, logical_types.get(time_index))
    if semantic_tags:
        _check_semantic_tags(column_names, semantic_tags)
    if column_descriptions:
        _check_column_descriptions(column_names, column_descriptions)


def _check_column_names(column_names):
    if not isinstance(column_names, (list, set)):
        raise TypeError('Column names must be a list or set')

    if len(column_names) != len(set(column_names)):
        raise IndexError('Schema cannot contain duplicate columns names')


def _check_index(column_names, index):
    if index not in column_names:
        # User specifies an index that is not in the list of column names
        raise LookupError(f'Specified index column `{index}` not found in Schema.')


def _check_time_index(column_names, time_index, logical_type):
    if time_index not in column_names:
        raise LookupError(f'Specified time index column `{time_index}` not found in Schema')
    ltype_class = _get_ltype_class(logical_type)

    if not (ltype_class == ww.logical_types.Datetime or 'numeric' in ltype_class.standard_tags):
        raise TypeError('Time index column must be a Datetime or numeric column.')


def _check_logical_types(column_names, logical_types, require_all_cols=True):
    if not isinstance(logical_types, dict):
        raise TypeError('logical_types must be a dictionary')
    cols_in_ltypes = set(logical_types.keys())
    cols_in_schema = set(column_names)

    cols_not_found_in_schema = cols_in_ltypes.difference(cols_in_schema)
    if cols_not_found_in_schema:
        raise LookupError('logical_types contains columns that are not present in '
                          f'Schema: {sorted(list(cols_not_found_in_schema))}')
    cols_not_found_in_ltypes = cols_in_schema.difference(cols_in_ltypes)
    if cols_not_found_in_ltypes and require_all_cols:
        raise LookupError(f'logical_types is missing columns that are present in '
                          f'Schema: {sorted(list(cols_not_found_in_ltypes))}')

    for col_name, logical_type in logical_types.items():
        if _get_ltype_class(logical_type) not in ww.type_system.registered_types:
            raise TypeError("Logical Types must be of the LogicalType class "
                            "and registered in Woodwork's type system. "
                            f"{logical_type} does not meet that criteria.")


def _check_semantic_tags(column_names, semantic_tags):
    if not isinstance(semantic_tags, dict):
        raise TypeError('semantic_tags must be a dictionary')
    cols_not_found = set(semantic_tags.keys()).difference(set(column_names))
    if cols_not_found:
        raise LookupError('semantic_tags contains columns that do not exist: '
                          f'{sorted(list(cols_not_found))}')

    for col_name, col_tags in semantic_tags.items():
        if not isinstance(col_tags, (str, list, set)):
            raise TypeError(f"semantic_tags for {col_name} must be a string, set or list")


def _check_column_descriptions(column_names, column_descriptions):
    if not isinstance(column_descriptions, dict):
        raise TypeError('column_descriptions must be a dictionary')
    cols_not_found = set(column_descriptions.keys()).difference(set(column_names))
    if cols_not_found:
        raise LookupError('column_descriptions contains columns that do not exist: '
                          f'{sorted(list(cols_not_found))}')


def _check_table_metadata(table_metadata):
    if not isinstance(table_metadata, dict):
        raise TypeError('Table metadata must be a dictionary.')


def _check_column_metadata(column_names, column_metadata):
    if not isinstance(column_metadata, dict):
        raise TypeError('Column metadata must be a dictionary.')
    cols_not_found = set(column_metadata.keys()).difference(set(column_names))
    if cols_not_found:
        raise LookupError('column_metadata contains columns that do not exist: '
                          f'{sorted(list(cols_not_found))}')


def _validate_not_setting_index_tags(semantic_tags, col_name):
    """Verify user has not supplied tags that cannot be set directly"""
    if 'index' in semantic_tags:
        raise ValueError(f"Cannot add 'index' tag directly for column {col_name}. To set a column as the index, "
                         "use DataFrame.ww.set_index() instead.")
    if 'time_index' in semantic_tags:
        raise ValueError(f"Cannot add 'time_index' tag directly for column {col_name}. To set a column as the time index, "
                         "use DataFrame.ww.set_time_index() instead.")
