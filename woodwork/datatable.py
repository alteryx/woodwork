import warnings

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

import woodwork as ww
import woodwork.serialize as serialize
from woodwork.datacolumn import DataColumn
from woodwork.exceptions import ColumnNameMismatchWarning
from woodwork.indexers import _iLocIndexer
from woodwork.logical_types import Boolean, Datetime, Double, LatLong
from woodwork.type_sys.utils import (
    _get_ltype_class,
    _is_numeric_series,
    col_is_datetime
)
from woodwork.utils import (
    _convert_input_to_set,
    _get_mode,
    _new_dt_including,
    import_or_none,
    import_or_raise
)

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')
if ks:
    ks.set_option('compute.ops_on_diff_frames', True)


class DataTable(object):
    def __init__(self, dataframe,
                 name=None,
                 index=None,
                 time_index=None,
                 semantic_tags=None,
                 logical_types=None,
                 table_metadata=None,
                 column_metadata=None,
                 use_standard_tags=True,
                 make_index=False,
                 column_descriptions=None,
                 already_sorted=False):
        """Create DataTable

        Args:
            dataframe (pd.DataFrame, dd.DataFrame, ks.DataFrame, numpy.ndarray): Dataframe providing the data for the datatable.
            name (str, optional): Name used to identify the datatable.
            index (str, optional): Name of the index column in the dataframe.
            time_index (str, optional): Name of the time index column in the dataframe.
            semantic_tags (dict, optional): Dictionary mapping column names in the dataframe to the
                semantic tags for the column. The keys in the dictionary should be strings
                that correspond to columns in the underlying dataframe. There are two options for
                specifying the dictionary values:
                (str): If only one semantic tag is being set, a single string can be used as a value.
                (list[str] or set[str]): If multiple tags are being set, a list or set of strings can be
                used as the value.
                Semantic tags will be set to an empty set for any column not included in the
                dictionary.
            logical_types (dict[str -> LogicalType], optional): Dictionary mapping column names in
                the dataframe to the LogicalType for the column. LogicalTypes will be inferred
                for any columns not present in the dictionary.
            table_metadata (dict[str -> json serializable], optional): Dictionary containing extra metadata for the DataTable.
            column_metadata (dict[str -> dict[str -> json serializable]], optional): Dictionary mapping column names
                to that column's metadata dictionary.
            use_standard_tags (bool, optional): If True, will add standard semantic tags to columns based
                on the inferred or specified logical type for the column. Defaults to True.
            make_index (bool, optional): If True, will create a new unique, numeric index column with the
                name specified by ``index`` and will add the new index column to the supplied DataFrame.
                If True, the name specified in ``index`` cannot match an existing column name in
                ``dataframe``. If False, the name is specified in ``index`` must match a column
                present in the ``dataframe``. Defaults to False.
            column_descriptions (dict[str -> str], optional): Dictionary containing column descriptions
            already_sorted (bool, optional): Indicates whether the input dataframe is already sorted on the time
                index. If False, will sort the dataframe first on the time_index and then on the index (pandas DataFrame
                only). Defaults to False.
        """
        # Check that inputs are valid
        dataframe = _validate_dataframe(dataframe)
        _validate_params(dataframe, name, index, time_index, logical_types,
                         table_metadata, column_metadata, semantic_tags, make_index, column_descriptions)

        self._dataframe = dataframe

        self.make_index = make_index or None
        if self.make_index:
            self._dataframe = _make_index(self._dataframe, index)

        self.name = name
        self.use_standard_tags = use_standard_tags

        # Infer logical types and create columns
        self.columns = self._create_columns(self._dataframe.columns,
                                            logical_types,
                                            semantic_tags,
                                            use_standard_tags,
                                            column_descriptions,
                                            column_metadata)
        if index is not None:
            _update_index(self, index)

        # Update dtypes before setting time index so that any Datetime formatting is applied
        self._update_columns(self.columns)

        needs_index_update = self._set_underlying_index()

        needs_sorting_update = False
        if time_index is not None:
            _update_time_index(self, time_index)
            needs_sorting_update = not self._sort_columns(already_sorted)

        if needs_index_update or needs_sorting_update:
            self._update_columns_from_dataframe()

        self.metadata = table_metadata or {}

    def __eq__(self, other, deep=True):
        if self.name != other.name:
            return False
        if self.index != other.index:
            return False
        if self.time_index != other.time_index:
            return False
        if set(self.columns.keys()) != set(other.columns.keys()):
            return False
        if self.metadata != other.metadata:
            return False
        for col_name in self.columns.keys():
            if self[col_name] != other[col_name]:
                return False

        # Only check pandas DataFrames for equality
        if isinstance(self.to_dataframe(), pd.DataFrame) and isinstance(other.to_dataframe(), pd.DataFrame):
            return self.to_dataframe().equals(other.to_dataframe())
        return True

    def __getitem__(self, key):
        if isinstance(key, list):
            invalid_cols = set(key).difference(set(self.columns.keys()))
            if invalid_cols:
                raise KeyError(f"Column(s) '{', '.join(sorted(list(invalid_cols)))}' not found in DataTable")
            return self._new_dt_from_cols(key)
        if key not in self.columns.keys():
            raise KeyError(f"Column with name {key} not found in DataTable")
        return self.columns[key]

    def __setitem__(self, col_name, column):
        if not isinstance(column, DataColumn):
            raise ValueError('New column must be of DataColumn type')

        # Don't allow reassigning of index or time index with setitem
        if self.index == col_name:
            raise KeyError('Cannot reassign index. Change column name and then use dt.set_index to reassign index.')
        if self.time_index == col_name:
            raise KeyError('Cannot reassign time index. Change column name and then use dt.set_time_index to reassign time index.')

        if column.name is not None and column.name != col_name:
            warnings.warn(ColumnNameMismatchWarning().get_warning_message(column.name, col_name),
                          ColumnNameMismatchWarning)
        column._series.name = col_name
        column._assigned_name = col_name

        self._dataframe[col_name] = column._series
        self._update_columns({col_name: column})

    def __sizeof__(self):
        return self._dataframe.__sizeof__()

    def __len__(self):
        return self._dataframe.__len__()

    def __repr__(self):
        '''A string representation of a DataTable containing typing information and a preview of the data.
        '''
        dt_repr = self._get_typing_info()
        if isinstance(dt_repr, str):
            return dt_repr

        return repr(dt_repr)

    def _repr_html_(self):
        '''An HTML representation of a DataTable for IPython.display in Jupyter Notebooks
        containing typing information and a preview of the data.
        '''
        dt_repr = self._get_typing_info()
        if isinstance(dt_repr, str):
            return dt_repr

        return dt_repr.to_html()

    @property
    def types(self):
        """Dataframe containing the physical dtypes, logical types and semantic
        tags for the table"""
        return self._get_typing_info()

    def head(self, n=5):
        '''Shows the first n rows of the DataTable along with typing information.

        Note:
            This will bring data into memory for Dask or Koalas DataTables.

        Args:
            n (int): number of rows to return. Defaults to 5.

        Returns:
            DataFrame with the top n rows where the column headers contain
        each DataColumn's typing information.
        '''

        typing_info = self._get_typing_info(include_names_col=True)
        if isinstance(typing_info, str):
            return typing_info

        data = self._dataframe.head(n)
        if not isinstance(data, pd.DataFrame):
            data = data.to_pandas()
        data.columns = pd.MultiIndex.from_frame(typing_info)

        return data

    def _get_typing_info(self, include_names_col=False):
        '''Creates a DataFrame that contains the typing information for a DataTable,
        optionally including the Data Column names as a column in addition to being
        the index.
        '''
        if len(self._dataframe.index) == 0 and len(self._dataframe.columns) == 0:
            return "Empty DataTable"

        typing_info = {}
        # Access column names from underlying data to maintain column order
        for col_name in self._dataframe.columns:
            dc = self[col_name]
            types = [dc.dtype, dc.logical_type, str(list(dc.semantic_tags))]
            if include_names_col:
                types.insert(0, dc.name)
            typing_info[dc.name] = types

        columns = ['Physical Type', 'Logical Type', 'Semantic Tag(s)']
        index = 'Data Column'
        if include_names_col:
            columns.insert(0, index)

        df = pd.DataFrame.from_dict(typing_info,
                                    orient='index',
                                    columns=columns,
                                    dtype="object")
        df.index.name = index
        return df

    @property
    def ltypes(self):
        """A series listing the logical types for each column in the table"""
        return self.types['Logical Type']

    def _create_columns(self,
                        column_names,
                        logical_types,
                        semantic_tags,
                        use_standard_tags,
                        column_descriptions,
                        column_metadata):
        """Create a dictionary with column names as keys and new DataColumn objects
        as values, while assigning any values that are passed for logical types or
        semantic tags to the new column."""
        datacolumns = {}
        for name in column_names:
            if logical_types and name in logical_types:
                logical_type = logical_types[name]
            else:
                logical_type = None
            if semantic_tags and name in semantic_tags:
                semantic_tag = semantic_tags[name]
            else:
                semantic_tag = None
            if column_descriptions:
                description = column_descriptions.get(name)
            else:
                description = None
            if column_metadata:
                metadata = column_metadata.get(name)
            else:
                metadata = None
            dc = DataColumn(self._dataframe[name], logical_type, semantic_tag, use_standard_tags, name, description, metadata)
            datacolumns[dc.name] = dc
        return datacolumns

    @property
    def logical_types(self):
        """A dictionary containing logical types for each column"""
        return {dc.name: dc.logical_type for dc in self.columns.values()}

    @property
    def physical_types(self):
        """A dictionary containing physical types for each column"""
        return {dc.name: dc.dtype for dc in self.columns.values()}

    @property
    def semantic_tags(self):
        """A dictionary containing semantic tags for each column"""
        return {dc.name: dc.semantic_tags for dc in self.columns.values()}

    @property
    def shape(self):
        """Returns a tuple representing the dimensionality of the DataTable. If Dask DataFrame, returns
            a Dask `Delayed` object for the number of rows."""
        return self._dataframe.shape

    @property
    def index(self):
        """The index column for the table"""
        for column in self.columns.values():
            if 'index' in column.semantic_tags:
                return column.name
        return None

    @index.setter
    def index(self, index):
        if self.index is not None and index is None:
            updated_index_col = self.columns[self.index].remove_semantic_tags('index')
            self._update_columns({self.index: updated_index_col})
        elif index is not None:
            _update_index(self, index, self.index)
        # Update the underlying index
        needs_update = self._set_underlying_index()
        if needs_update:
            self._update_columns_from_dataframe()

    @property
    def time_index(self):
        """The time index column for the table"""
        for column in self.columns.values():
            if 'time_index' in column.semantic_tags:
                return column.name
        return None

    @time_index.setter
    def time_index(self, time_index):
        if self.time_index is not None and time_index is None:
            updated_time_index_col = self.columns[self.time_index].remove_semantic_tags('time_index')
            self._update_columns({self.time_index: updated_time_index_col})
        elif time_index is not None:
            _update_time_index(self, time_index, self.time_index)

    def _update_columns(self, new_columns):
        """Update the DataTable columns based on items contained in the
        provided new_columns dictionary"""
        for name, column in new_columns.items():
            self.columns[name] = column
            # Make sure the underlying dataframe is in sync in case series data has changed
            self._dataframe[name] = column._series

    def _update_columns_from_dataframe(self):
        '''
        Update each DataColumns' series based on the current DataTable's dataframe
        '''
        for column in self.columns.keys():
            self.columns[column]._set_series(self._dataframe[column])

    def _sort_columns(self, already_sorted):
        if dd and isinstance(self._dataframe, dd.DataFrame) or (ks and isinstance(self._dataframe, ks.DataFrame)):
            already_sorted = True  # Skip sorting for Dask and Koalas input
        if not already_sorted:
            sort_cols = [self.time_index, self.index]
            if self.index is None:
                sort_cols = [self.time_index]
            self._dataframe = self._dataframe.sort_values(sort_cols)

        return already_sorted

    def _set_underlying_index(self):
        '''Sets the index of a DataTable's underlying dataframe on pandas DataTables.

        If the DataTable has an index, will be set to that index.
        If no index is specified and the DataFrame's index isn't a RangeIndex, will reset the DataFrame's index,
        meaning that the index will be a pd.RangeIndex starting from zero.
        '''
        needs_update = False
        new_df = self._dataframe
        if isinstance(self._dataframe, pd.DataFrame):
            if self.index is not None:
                needs_update = True
                new_df = self._dataframe.set_index(self.index, drop=False)
                # Drop index name to not overlap with the original column
                new_df.index.name = None
            # Only reset the index if the index isn't a RangeIndex
            elif not isinstance(self._dataframe.index, pd.RangeIndex):
                needs_update = True
                new_df = self._dataframe.reset_index(drop=True)

        self._dataframe = new_df
        return needs_update

    def pop(self, column_name):
        """Return a DataColumn and drop it from the DataTable.

        Args:
            column (str): Name of the column to pop.

        Returns:
            woodwork.DataColumn: DataColumn including logical type and semantic tags.
        """
        col = self[column_name]
        del self.columns[column_name]
        self._dataframe = self._dataframe.drop(column_name, axis=1)
        return col

    def drop(self, columns):
        """Drop specified columns from a DataTable.

        Args:
            columns (str or list[str]): Column name or names to drop. Must be present in the DataTable.

        Returns:
            woodwork.DataTable: DataTable with the specified columns removed.
        """
        if not isinstance(columns, (list, set)):
            columns = [columns]

        not_present = [col for col in columns if col not in self.columns]
        if not_present:
            raise ValueError(f'{not_present} not found in DataTable')

        return self._new_dt_from_cols([col for col in self._dataframe.columns if col not in columns])

    def rename(self, columns):
        """Renames columns in a DataTable

        Args:
            columns (dict[str -> str]): A dictionary mapping columns whose names
                we'd like to change to the name to which we'd like to change them.

        Returns:
            woodwork.DataTable: DataTable with the specified columns renamed.

        Note:
            Index and time index columns cannot be renamed.
        """
        for old_name, new_name in columns.items():
            if old_name not in self.columns:
                raise KeyError(f"Column to rename must be present in the DataTable. {old_name} is not present in the DataTable.")
            if new_name in self.columns and new_name not in columns.keys():
                raise ValueError(f"The column {new_name} is already present in the DataTable. Please choose another name to rename {old_name} to or also rename {old_name}.")
            if old_name == self.index or old_name == self.time_index:
                raise KeyError(f"Cannot rename index or time index columns such as {old_name}.")

        if len(columns) != len(set(columns.values())):
            raise ValueError('New columns names must be unique from one another.')

        old_all_cols = list(self._dataframe.columns)
        new_dt = self._new_dt_from_cols(old_all_cols)
        updated_cols = {}
        for old_name, new_name in columns.items():
            col = new_dt.pop(old_name)

            col._series.name = new_name
            col._assigned_name = new_name
            updated_cols[new_name] = col

        new_dt._update_columns(updated_cols)
        # Make sure we maintain relative order of columns including new names
        new_all_cols = [name if name not in columns else columns[name] for name in old_all_cols]
        new_dt._dataframe = new_dt._dataframe[new_all_cols]

        return new_dt

    def set_index(self, index):
        """Set the index column and return a new DataTable. Adds the 'index' semantic
        tag to the column and clears the tag from any previously set index column.
        Setting a column as the index column will also cause any previously set standard
        tags for the column to be removed.

        Args:
            index (str): The name of the column to set as the index

        Returns:
            woodwork.DataTable: DataTable with the specified index column set.
        """
        new_dt = self._new_dt_from_cols(self._dataframe.columns)
        _update_index(new_dt, index, self.index)
        needs_update = new_dt._set_underlying_index()
        if needs_update:
            self._update_columns_from_dataframe()
        return new_dt

    def set_time_index(self, time_index):
        """Set the time index column. Adds the 'time_index' semantic tag to the column and
        clears the tag from any previously set index column

        Args:
            time_index (str): The name of the column to set as the time index.
        """
        new_dt = self._new_dt_from_cols(self._dataframe.columns)
        _update_time_index(new_dt, time_index, self.time_index)
        return new_dt

    def set_types(self, logical_types=None, semantic_tags=None, retain_index_tags=True):
        """Update the logical type and semantic tags for any columns names in the provided types
        dictionary. Replaces existing columns with new DataColumn objects and returns a new
        DataTable object.

        Args:
            logical_types (dict[str -> str], optional): A dictionary defining the new logical types for the
                specified columns.
            semantic_tags (dict[str -> str/list/set], optional): A dictionary defining the new semantic_tags for the
                specified columns.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will replace all semantic tags. Defaults to
                True.

        Returns:
            woodwork.DataTable: DataTable with updated logical types and specified semantic tags set.
        """
        logical_types = logical_types or {}
        _check_logical_types(self._dataframe, logical_types)

        semantic_tags = semantic_tags or {}
        _check_semantic_tags(self._dataframe, semantic_tags)

        new_dt = self._new_dt_from_cols(self._dataframe.columns)
        cols_to_update = {}
        for col_name, col in new_dt.columns.items():
            if col_name not in logical_types and col_name not in semantic_tags:
                continue
            if col_name in logical_types:
                col = col.set_logical_type(logical_types[col_name], retain_index_tags)
            if col_name in semantic_tags:
                col = col.set_semantic_tags(semantic_tags[col_name], retain_index_tags)
            cols_to_update[col_name] = col

        new_dt._update_columns(cols_to_update)
        return new_dt

    def add_semantic_tags(self, semantic_tags):
        """Adds specified semantic tags to columns. Will retain any previously set values.
        Replaces updated columns with new DataColumn objects and returns a new DataTable object.

        Args:
            semantic_tags (dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataTable to the tags that should be added to the column

        Returns:
            woodwork.DataTable: DataTable with semantic tags added
        """
        _check_semantic_tags(self._dataframe, semantic_tags)
        return self._update_cols_and_get_new_dt('add_semantic_tags', semantic_tags)

    def remove_semantic_tags(self, semantic_tags):
        """Remove the semantic tags for any column names in the provided semantic_tags
        dictionary. Replaces the column with a new DataColumn object and return a new DataTable
        object.

        Args:
            semantic_tags (dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataTable to the tags that should be removed to the column

        Returns:
            woodwork.DataTable: DataTable with the specified semantic tags removed
        """
        _check_semantic_tags(self._dataframe, semantic_tags)
        return self._update_cols_and_get_new_dt('remove_semantic_tags', semantic_tags)

    def reset_semantic_tags(self, columns=None, retain_index_tags=False):
        """Reset the semantic tags for the specified columns to the default values and
        return a new DataTable. The default values will be either an empty set or a set
        of the standard tags based on the column logical type, controlled by the
        use_standard_tags property on the table. Columns names can be provided as a
        single string, a list of strings or a set of strings. If columns is not specified,
        tags will be reset for all columns.

        Args:
            columns (str/list/set): The columns for which the semantic tags should be reset.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will clear all semantic tags. Defaults to
                False.

        Returns:
            woodwork.DataTable: DataTable with semantic tags reset to default values
        """
        columns = _convert_input_to_set(columns, "columns")
        cols_not_found = sorted(list(columns.difference(set(self._dataframe.columns))))
        if cols_not_found:
            raise LookupError("Input contains columns that are not present in "
                              f"dataframe: '{', '.join(cols_not_found)}'")
        if not columns:
            columns = self._dataframe.columns
        return self._update_cols_and_get_new_dt('reset_semantic_tags', columns, retain_index_tags)

    def _update_cols_and_get_new_dt(self, method, new_values, *args):
        """Helper method that can be used for updating columns by calling the column method
        that is specified, passing along information contained in new_values and any
        additional positional arguments.

        Args:
            method (str): The name of the method to call on the DataColumn object to perform the update.
            new_values (dict/list): If a dictionary is provided the keys should correspond to column
                names and the items in the dictionary values will be passed along to the DataColumn method.
                If a list is provided, the items in the list should correspond to column names and
                no additional values will be passed along to the DataColumn method.

        Returns:
            woodwork.DataTable: A new DataTable with updated columns
        """
        new_dt = self._new_dt_from_cols(self._dataframe.columns)
        cols_to_update = {}
        if isinstance(new_values, dict):
            for name, tags in new_values.items():
                cols_to_update[name] = getattr(new_dt.columns[name], method)(tags, *args)
        else:
            for name in new_values:
                cols_to_update[name] = getattr(new_dt.columns[name], method)(*args)
        new_dt._update_columns(cols_to_update)
        return new_dt

    def to_dataframe(self):
        """Retrieves the DataTable's underlying dataframe.

        Note: Do not modify the returned dataframe directly to avoid unexpected behavior

        Returns:
            DataFrame: The underlying dataframe of the DataTable. Return type will depend on the type
                of dataframe used to create the DataTable.
        """
        return self._dataframe

    @property
    def df(self):
        return self.to_dataframe()

    def select(self, include):
        """Create a DataTable including only columns whose logical type and
        semantic tags are specified in the list of types and tags to include.
        If no matching columns are found, an empty DataTable will be returned.

        Args:
            include (str or LogicalType or list[str or LogicalType]): Logical
                types, semantic tags or column names to include
                in the DataTable.

        Returns:
            DataTable: The subset of the original DataTable that contains just the
            logical types and semantic tags in ``include``.
        """
        cols_to_include = self._filter_cols(include)
        return self._new_dt_from_cols(cols_to_include)

    def update_dataframe(self, new_df, already_sorted=False):
        '''Replace the DataTable's dataframe with a new dataframe, making sure the new dataframe dtypes are updated.
        If the original DataTable was created with ``make_index=True``, an index column will be added to the updated
        data if it is not present.

        Args:
            new_df (DataFrame): Dataframe containing the new data. The same columns present in the original data should
                also be present in the new dataframe.
            already_sorted (bool, optional): Indicates whether the input dataframe is already sorted on the time
                index. If False, will sort the dataframe first on the time_index and then on the index (pandas DataFrame
                only). Defaults to False.
        '''
        if self.make_index and self.index not in new_df.columns:
            new_df = _make_index(new_df, self.index)

        if len(new_df.columns) != len(self.columns):
            raise ValueError("Updated dataframe contains {} columns, expecting {}".format(len(new_df.columns),
                                                                                          len(self.columns)))
        for column in self.columns.keys():
            if column not in new_df.columns:
                raise ValueError("Updated dataframe is missing new {} column".format(column))

        if self.index:
            _check_index(new_df, self.index)

        # Make sure column ordering matches existing ordering
        new_df = new_df[[column for column in self._dataframe.columns]]
        self._dataframe = new_df

        if self.time_index is not None:
            _check_time_index(new_df, self.time_index)

        # Set underlying index and sort on it, if necessary
        self._set_underlying_index()
        if self.time_index is not None:
            self._sort_columns(already_sorted)

        # Update column series and dtype
        for column in self.columns.keys():
            self.columns[column]._set_series(self._dataframe[column])
            self.columns[column]._update_dtype()

        # Make sure dataframe dtypes match columns
        self._update_columns(self.columns)

    def _filter_cols(self, include, col_names=False):
        """Return list of columns filtered in specified way. In case of collision, favors logical types
        then semantic tag then column name.

        Args:
            include (str or LogicalType or list[str or LogicalType]): parameter or list of parameters to
                filter columns by.

            col_names (bool): Specifies whether to filter columns by name. Defaults to False.

        Returns:
            List[str] of column names that fit into filter.
        """
        if not isinstance(include, list):
            include = [include]

        ltypes_used = set()
        ltypes_in_dt = {_get_ltype_class(col.logical_type) for col in self.columns.values()}

        tags_used = set()
        tags_in_dt = {tag for col in self.columns.values() for tag in col.semantic_tags}

        cols_to_include = set()

        for selector in include:
            if _get_ltype_class(selector) in ww.type_system.registered_types:
                if selector not in ww.type_system.registered_types:
                    raise TypeError(f"Invalid selector used in include: {selector} cannot be instantiated")
                if selector in ltypes_in_dt:
                    ltypes_used.add(selector)
            elif isinstance(selector, str):
                # If the str is a viable ltype, it'll take precedence
                # but if it's not present, we'll check if it's a tag
                ltype = ww.type_system.str_to_logical_type(selector, raise_error=False)
                if ltype and ltype in ltypes_in_dt:
                    ltypes_used.add(ltype)
                    continue
                elif selector in tags_in_dt:
                    tags_used.add(selector)
                elif col_names and selector in self.columns:
                    cols_to_include.add(selector)
            else:
                raise TypeError(f"Invalid selector used in include: {selector} must be either a string or LogicalType")

        for col_name, col in self.columns.items():
            if _get_ltype_class(col.logical_type) in ltypes_used or col.semantic_tags.intersection(tags_used):
                cols_to_include.add(col_name)

        # Maintain column order by using underlying data
        return [col_name for col_name in self._dataframe.columns if col_name in cols_to_include]

    def _new_dt_from_cols(self, cols_to_include):
        """Creates a new DataTable from a list of column names, retaining all types,
        indices, and name of original DataTable. Resulting DataTable's column order will
        follow the order used in cols_to_include."""
        assert all([col_name in self.columns for col_name in cols_to_include])
        return _new_dt_including(self, self._dataframe.loc[:, cols_to_include])

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position.
        ``.iloc[]`` is primarily integer position based (from ``0`` to
        ``length-1`` of the axis), but may also be used with a boolean array.

        Allowed inputs are:
            An integer, e.g. ``5``.
            A list or array of integers, e.g. ``[4, 3, 0]``.
            A slice object with ints, e.g. ``1:7``.
            A boolean array.
            A ``callable`` function with one argument (the calling Series, DataFrame
            or Panel) and that returns valid output for indexing (one of the above).
            This is useful in method chains, when you don't have a reference to the
            calling object, but would like to base your selection on some value.
        """
        return _iLocIndexer(self)

    def describe_dict(self, include=None):
        """Calculates statistics for data contained in DataTable.

        Args:
            include (list[str or LogicalType], optional): filter for what columns to include in the
            statistics returned. Can be a list of columns, semantic tags, logical types, or a list
            combining any of the three. It follows the most broad specification. Favors logical types
            then semantic tag then column name. If no matching columns are found, an empty DataFrame
            will be returned.

        Returns:
            dict[str -> dict]: A dictionary with a key for each column in the data or for each column
            matching the logical types, semantic tags or column names specified in ``include``, paired
            with a value containing a dictionary containing relevant statistics for that column.
        """
        agg_stats_to_calculate = {
            'category': ["count", "nunique"],
            'numeric': ["count", "max", "min", "nunique", "mean", "std"],
            Datetime: ["count", "max", "min", "nunique", "mean"],
        }
        if include is not None:
            filtered_cols = self._filter_cols(include, col_names=True)
            cols_to_include = [(k, v) for k, v in self.columns.items() if k in filtered_cols]
        else:
            cols_to_include = self.columns.items()

        results = {}

        if dd and isinstance(self._dataframe, dd.DataFrame):
            df = self._dataframe.compute()
        elif ks and isinstance(self._dataframe, ks.DataFrame):
            # Missing values in Koalas will be replaced with 'None' - change them to
            # np.nan so stats are calculated properly
            df = self._dataframe.to_pandas().replace(to_replace='None', value=np.nan)

            # Any LatLong columns will be using lists, which we must convert
            # back to tuples so we can calculate the mode, which requires hashable values
            latlong_columns = [col_name for col_name, col in self.columns.items() if _get_ltype_class(col.logical_type) == LatLong]
            df[latlong_columns] = df[latlong_columns].applymap(lambda latlong: tuple(latlong) if latlong else latlong)
        else:
            df = self._dataframe

        for column_name, column in cols_to_include:
            if 'index' in column.semantic_tags:
                continue
            values = {}
            logical_type = column.logical_type
            semantic_tags = column.semantic_tags
            series = df[column_name]

            # Calculate Aggregation Stats
            if column._is_categorical():
                agg_stats = agg_stats_to_calculate['category']
            elif column._is_numeric():
                agg_stats = agg_stats_to_calculate['numeric']
            elif _get_ltype_class(logical_type) == Datetime:
                agg_stats = agg_stats_to_calculate[Datetime]
            else:
                agg_stats = ["count"]
            values = series.agg(agg_stats).to_dict()

            # Calculate other specific stats based on logical type or semantic tags
            if _get_ltype_class(logical_type) == Boolean:
                values["num_false"] = series.value_counts().get(False, 0)
                values["num_true"] = series.value_counts().get(True, 0)
            elif column._is_numeric():
                quant_values = series.quantile([0.25, 0.5, 0.75]).tolist()
                values["first_quartile"] = quant_values[0]
                values["second_quartile"] = quant_values[1]
                values["third_quartile"] = quant_values[2]

            mode = _get_mode(series)
            # The format of the mode should match its format in the DataTable
            if ks and isinstance(self._dataframe, ks.DataFrame) and series.name in latlong_columns:
                mode = list(mode)

            values["nan_count"] = series.isna().sum()
            values["mode"] = mode
            values["physical_type"] = column.dtype
            values["logical_type"] = logical_type
            values["semantic_tags"] = semantic_tags
            results[column_name] = values
        return results

    def describe(self, include=None):
        """Calculates statistics for data contained in DataTable.

        Args:
            include (list[str or LogicalType], optional): filter for what columns to include in the
            statistics returned. Can be a list of columns, semantic tags, logical types, or a list
            combining any of the three. It follows the most broad specification. Favors logical types
            then semantic tag then column name. If no matching columns are found, an empty DataFrame
            will be returned.

        Returns:
            pd.DataFrame: A Dataframe containing statistics for the data or the subset of the original
            DataTable that contains the logical types, semantic tags, or column names specified
            in ``include``.
        """
        results = self.describe_dict(include=include)
        index_order = [
            'physical_type',
            'logical_type',
            'semantic_tags',
            'count',
            'nunique',
            'nan_count',
            'mean',
            'mode',
            'std',
            'min',
            'first_quartile',
            'second_quartile',
            'third_quartile',
            'max',
            'num_true',
            'num_false',
        ]
        return pd.DataFrame(results).reindex(index_order)

    def value_counts(self, ascending=False, top_n=10, dropna=False):
        """Returns a list of dictionaries with counts for the most frequent values in each column (only
            for DataColumns with `category` as a standard tag).


        Args:
            ascending (bool): Defines whether each list of values should be sorted most frequent
                to least frequent value (False), or least frequent to most frequent value (True).
                Defaults to False.

            top_n (int): the number of top values to retrieve. Defaults to 10.

            dropna (bool): determines whether to remove NaN values when finding frequency. Defaults
                to False.

        Returns:
            top_list (list(dict)): a list of dictionaries for each categorical column with keys `count`
                and `value`.
        """
        val_counts = {}
        valid_cols = [col for col, column in self.columns.items() if column._is_categorical()]
        data = self._dataframe[valid_cols]
        is_ks = False
        if dd and isinstance(data, dd.DataFrame):
            data = data.compute()
        if ks and isinstance(data, ks.DataFrame):
            data = data.to_pandas()
            is_ks = True

        for col in valid_cols:
            if dropna and is_ks:
                # Koalas categorical columns will have missing values replaced with the string 'None'
                # Replace them with np.nan so dropna work
                datacol = data[col].replace(to_replace='None', value=np.nan)
            else:
                datacol = data[col]
            frequencies = datacol.value_counts(ascending=ascending, dropna=dropna)
            df = frequencies[:top_n].reset_index()
            df.columns = ["value", "count"]
            dt_list = list(df.to_dict(orient="index").values())
            val_counts[col] = dt_list
        return val_counts

    def _handle_nans_for_mutual_info(self, data):
        """
        Remove NaN values in the dataframe so that mutual information can be calculated

        Args:
            data (pd.DataFrame): dataframe to use for calculating mutual information

        Returns:
            pd.DataFrame: data with fully null columns removed and nans filled in
                with either mean or mode

        """
        # remove fully null columns
        data = data.loc[:, data.columns[data.notnull().any()]]

        # replace or remove null values
        for column_name in data.columns[data.isnull().any()]:
            column = self[column_name]
            series = data[column_name]
            ltype = column._logical_type

            if column._is_numeric():
                mean = series.mean()
                if isinstance(mean, float) and not _get_ltype_class(ltype) == Double:
                    data[column_name] = series.astype('float')
                data[column_name] = series.fillna(mean)
            elif column._is_categorical() or _get_ltype_class(ltype) == Boolean:
                mode = _get_mode(series)
                data[column_name] = series.fillna(mode)
        return data

    def _make_categorical_for_mutual_info(self, data, num_bins):
        """Transforms dataframe columns into numeric categories so that
        mutual information can be calculated

        Args:
            data (pd.DataFrame): dataframe to use for caculating mutual information
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical.


        Returns:
            data (pd.DataFrame): Transformed data
        """

        col_names = data.columns.to_list()
        for col_name in col_names:
            if self[col_name]._is_numeric():
                # bin numeric features to make categories
                data[col_name] = pd.qcut(data[col_name], num_bins, duplicates="drop")
            # Convert Datetimes to total seconds - an integer - and bin
            if _get_ltype_class(self[col_name].logical_type) == Datetime:
                data[col_name] = pd.qcut(data[col_name].astype('int64'), num_bins, duplicates="drop")
            # convert categories to integers
            new_col = data[col_name]
            if str(new_col.dtype) != 'category':
                new_col = new_col.astype('category')
            data[col_name] = new_col.cat.codes
        return data

    def mutual_information_dict(self, num_bins=10, nrows=None):
        """
        Calculates mutual information between all pairs of columns in the DataTable that
        support mutual information. Logical Types that support mutual information are as
        follows:  Boolean, Categorical, CountryCode, Datetime, Double, Integer, Ordinal,
        SubRegionCode, and ZIPCode

        Args:
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical.
            nrows (int): The number of rows to sample for when determining mutual info.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.

        Returns:
            list(dict): A list containing dictionaries that have keys `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect dependency).
        """
        # We only want Numeric, Categorical, and Boolean columns
        # And we don't want the index column
        valid_columns = [col_name for col_name, column
                         in self.columns.items() if (col_name != self.index and
                                                     (column._is_numeric() or
                                                      column._is_categorical() or
                                                      _get_ltype_class(column.logical_type) == Boolean or
                                                      _get_ltype_class(column.logical_type) == Datetime)
                                                     )]

        data = self._dataframe[valid_columns]
        if dd and isinstance(data, dd.DataFrame):
            data = data.compute()
        if ks and isinstance(self._dataframe, ks.DataFrame):
            data = data.to_pandas()

        # cut off data if necessary
        if nrows is not None and nrows < data.shape[0]:
            data = data.sample(nrows)

        data = self._handle_nans_for_mutual_info(data)
        data = self._make_categorical_for_mutual_info(data, num_bins)

        # calculate mutual info for all pairs of columns
        mutual_info = []
        col_names = data.columns.to_list()
        for i, a_col in enumerate(col_names):
            for j in range(i, len(col_names)):
                b_col = col_names[j]
                if a_col == b_col:
                    # Ignore because the mutual info for a column with itself will always be 1
                    continue
                else:
                    mi_score = normalized_mutual_info_score(data[a_col], data[b_col])
                    mutual_info.append(
                        {"column_1": a_col, "column_2": b_col, "mutual_info": mi_score}
                    )
        mutual_info.sort(key=lambda mi: mi['mutual_info'], reverse=True)
        return mutual_info

    def mutual_information(self, num_bins=10, nrows=None):
        """
        Calculates mutual information between all pairs of columns in the DataTable that
        support mutual information. Logical Types that support mutual information are as
        follows:  Boolean, Categorical, CountryCode, Datetime, Double, Integer, Ordinal,
        SubRegionCode, and ZIPCode

        Args:
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical.
            nrows (int): The number of rows to sample for when determining mutual info.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.

        Returns:
            pd.DataFrame: A Dataframe containing mutual information with columns `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect dependency).
        """
        mutual_info = self.mutual_information_dict(num_bins, nrows)
        return pd.DataFrame(mutual_info)

    def to_dictionary(self):
        '''
        Get a DataTable's description

        Returns:
            description (dict) : Description of :class:`.DataTable`.
        '''
        return serialize.datatable_to_description(self)

    def to_csv(self, path, sep=',', encoding='utf-8', engine='python', compression=None, profile_name=None):
        '''Write DataTable to disk in the CSV format, location specified by `path`.
            Path could be a local path or a S3 path.
            If writing to S3 a tar archive of files will be written.

            Args:
                path (str) : Location on disk to write to (will be created as a directory)
                sep (str) : String of length 1. Field delimiter for the output file.
                encoding (str) : A string representing the encoding to use in the output file, defaults to 'utf-8'.
                engine (str) : Name of the engine to use. Possible values are: {'c', 'python'}.
                compression (str) : Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}.
                profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        '''
        serialize.write_datatable(self, path, format='csv', index=False,
                                  sep=sep, encoding=encoding, engine=engine,
                                  compression=compression, profile_name=profile_name)

    def to_pickle(self, path, compression=None, profile_name=None):
        '''Write DataTable to disk in the pickle format, location specified by `path`.
            Path could be a local path or a S3 path.
            If writing to S3 a tar archive of files will be written.

            Args:
                path (str) : Location on disk to write to (will be created as a directory)
                compression (str) : Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}.
                profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        '''
        serialize.write_datatable(self, path, format='pickle',
                                  compression=compression, profile_name=profile_name)

    def to_parquet(self, path, compression=None, profile_name=None):
        '''Write DataTable to disk in the parquet format, location specified by `path`.
            Path could be a local path or a S3 path.
            If writing to S3 a tar archive of files will be written.

            Note:
                As the engine `fastparquet` cannot handle nullable pandas dtypes, `pyarrow` will be used
                for serialization to parquet.

            Args:
                path (str): location on disk to write to (will be created as a directory)
                compression (str) : Name of the compression to use. Possible values are: {'snappy', 'gzip', 'brotli', None}.
                profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        '''
        import_error_message = (
            "The pyarrow library is required to serialize to parquet.\n"
            "Install via pip:\n"
            "    pip install pyarrow\n"
            "Install via conda:\n"
            "   conda install pyarrow -c conda-forge"
        )
        import_or_raise('pyarrow', import_error_message)
        serialize.write_datatable(self, path, format='parquet',
                                  engine='pyarrow', compression=compression,
                                  profile_name=profile_name)


def _validate_dataframe(dataframe):
    '''Check that the dataframe supplied during DataTable initialization is valid,
    and convert numpy array to pandas DataFrame if necessary.'''
    if not ((dd and isinstance(dataframe, dd.DataFrame)) or
            (ks and isinstance(dataframe, ks.DataFrame)) or
            isinstance(dataframe, (pd.DataFrame, np.ndarray))):
        raise TypeError('Dataframe must be one of: pandas.DataFrame, dask.DataFrame, koalas.DataFrame, numpy.ndarray')

    if isinstance(dataframe, np.ndarray):
        dataframe = pd.DataFrame(dataframe)
    return dataframe


def _validate_params(dataframe, name, index, time_index, logical_types,
                     table_metadata, column_metadata, semantic_tags,
                     make_index, column_descriptions):
    """Check that values supplied during DataTable initialization are valid"""
    _check_unique_column_names(dataframe)
    if name and not isinstance(name, str):
        raise TypeError('DataTable name must be a string')
    if index is not None or make_index:
        _check_index(dataframe, index, make_index)
    if logical_types:
        _check_logical_types(dataframe, logical_types)
    if table_metadata:
        _check_table_metadata(table_metadata)
    if column_metadata:
        _check_column_metadata(dataframe, column_metadata)
    if time_index is not None:
        datetime_format = None
        logical_type = None
        if logical_types is not None and time_index in logical_types:
            logical_type = logical_types[time_index]
            if _get_ltype_class(logical_types[time_index]) == Datetime:
                datetime_format = logical_types[time_index].datetime_format

        _check_time_index(dataframe, time_index, datetime_format=datetime_format, logical_type=logical_type)

    if semantic_tags:
        _check_semantic_tags(dataframe, semantic_tags)

    if column_descriptions:
        _check_column_descriptions(dataframe, column_descriptions)


def _check_unique_column_names(dataframe):
    if not dataframe.columns.is_unique:
        raise IndexError('Dataframe cannot contain duplicate columns names')


def _check_index(dataframe, index, make_index=False):
    if not make_index and index not in dataframe.columns:
        # User specifies an index that is not in the dataframe, without setting make_index to True
        raise LookupError(f'Specified index column `{index}` not found in dataframe. To create a new index column, set make_index to True.')
    if index is not None and not make_index and isinstance(dataframe, pd.DataFrame) and not dataframe[index].is_unique:
        # User specifies an index that is in the dataframe but not unique
        # Does not check for Dask as Dask does not support is_unique
        raise IndexError('Index column must be unique')
    if make_index and index is not None and index in dataframe.columns:
        # User sets make_index to True, but supplies an index name that matches a column already present
        raise IndexError('When setting make_index to True, the name specified for index cannot match an existing column name')
    if make_index and index is None:
        # User sets make_index to True, but does not supply a name for the index
        raise IndexError('When setting make_index to True, the name for the new index must be specified in the index parameter')


def _check_time_index(dataframe, time_index, datetime_format=None, logical_type=None):
    if time_index not in dataframe.columns:
        raise LookupError(f'Specified time index column `{time_index}` not found in dataframe')
    if not (_is_numeric_series(dataframe[time_index], logical_type) or
            col_is_datetime(dataframe[time_index], datetime_format=datetime_format)):
        raise TypeError('Time index column must contain datetime or numeric values')


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


def _check_column_descriptions(dataframe, column_descriptions):
    if not isinstance(column_descriptions, dict):
        raise TypeError('column_descriptions must be a dictionary')
    cols_not_found = set(column_descriptions.keys()).difference(set(dataframe.columns))
    if cols_not_found:
        raise LookupError('column_descriptions contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')


def _check_table_metadata(table_metadata):
    if not isinstance(table_metadata, dict):
        raise TypeError('Table metadata must be a dictionary.')


def _check_column_metadata(dataframe, column_metadata):
    if not isinstance(column_metadata, dict):
        raise TypeError('Column metadata must be a dictionary.')
    cols_not_found = set(column_metadata.keys()).difference(set(dataframe.columns))
    if cols_not_found:
        raise LookupError('column_metadata contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')


def _update_index(datatable, index, old_index=None):
    """Add the `index` tag to the specified index column and remove the tag from the
    old_index column, if specified. Also checks that the specified index column
    can be used as an index."""
    _check_index(datatable._dataframe, index)
    if old_index is not None:
        datatable._update_columns({old_index: datatable.columns[old_index].remove_semantic_tags('index')})
    datatable.columns[index]._set_as_index()


def _update_time_index(datatable, time_index, old_time_index=None):
    """Add the `time_index` tag to the specified time_index column and remove the tag from the
    old_time_index column, if specified. Also checks that the specified time_index
    column can be used as a time index."""

    _check_time_index(datatable._dataframe, time_index)
    if old_time_index is not None:
        datatable._update_columns({old_time_index: datatable.columns[old_time_index].remove_semantic_tags('time_index')})
    datatable.columns[time_index]._set_as_time_index()


def _make_index(dataframe, index):
    if dd and isinstance(dataframe, dd.DataFrame):
        dataframe[index] = 1
        dataframe[index] = dataframe[index].cumsum() - 1
    elif ks and isinstance(dataframe, ks.DataFrame):
        dataframe = dataframe.koalas.attach_id_column('distributed-sequence', index)
    else:
        dataframe.insert(0, index, range(len(dataframe)))

    return dataframe
