import warnings

import numpy as np
import pandas as pd

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
    import_or_none
)

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')
if ks:
    ks.set_option('compute.ops_on_diff_frames', True)


class Schema(object):
    def __init__(self):
        pass

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
        """Create Schema

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

        # --> remove this and determine passing in DataFrame to all the fns???
        self._dataframe = dataframe

        self.make_index = make_index or None
        if self.make_index:
            dataframe = _make_index(self._dataframe, index)

        self.name = name
        self.use_standard_tags = use_standard_tags

        # Infer logical types and create columns
        #  --> have a way of creating columns that stores info and infers types and updates df
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
