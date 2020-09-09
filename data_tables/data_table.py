import pandas as pd

from data_tables.data_column import DataColumn


class DataTable(object):
    def __init__(self, dataframe,
                 name=None,
                 index=None,
                 time_index=None,
                 semantic_types=None,
                 logical_types=None,
                 copy_dataframe=False):
        """ Create DataTable

        Args:
            dataframe (pd.DataFrame): Dataframe providing the data for the datatable.
            name (str, optional): Name used to identify the datatable.
            index (str, optional): Name of the index column in the dataframe.
            time_index (str, optional): Name of the time index column in the dataframe.
            semantic_types (dict, optional): Dictionary mapping column names in the dataframe to the
                semantic types for the column. The keys in the dictionary should be strings
                that correspond to columns in the underlying dataframe.

                There are several options for specifying the dictionary values:
                    (str) If no aditional data is needed and only one semantic type is being set, a single
                    string can be used as a value.

                    (list) If muliple types are being set and none require additional data, a list of strings
                    can be used as the value.

                    (dict) For columns that require additional data, a dictionary should be passed as
                    the value. In this dictionary, the keys should be strings corresponding to the type name
                    and the values should be a dictionary containing any additional data, or `None` if no
                    additional data is being set for a particular semantic type.

                Semantic types will be set to an empty dictionary for any column not included in the
                dictionary.
            logical_types (dict[str -> LogicalType], optional): Dictionary mapping column names in
                the dataframe to the LogicalType for the column. LogicalTypes will be inferred
                for any columns not present in the dictionary.
            copy_dataframe (bool, optional): If True, a copy of the input dataframe will be made
                prior to creating the DataTable. Defaults to False, which results in using a
                reference to the input dataframe.
        """
        # Check that inputs are valid
        _validate_params(dataframe, name, index, time_index, logical_types, semantic_types)

        if copy_dataframe:
            self.dataframe = dataframe.copy()
        else:
            self.dataframe = dataframe

        self.name = name
        self.index = index
        self.time_index = time_index

        # Infer logical types and create columns
        self.columns = self._create_columns(self.dataframe.columns,
                                            logical_types,
                                            semantic_types)

    @property
    def types(self):
        typing_info = {}
        for dc in self.columns.values():
            typing_info[dc.name] = [dc.dtype, dc.logical_type, dc.semantic_types]
        df = pd.DataFrame.from_dict(typing_info,
                                    orient='index',
                                    columns=['Physical Type', 'Logical Type', 'Semantic Tag(s)'],
                                    dtype="object")
        df.index.name = 'Data Column'
        return df

    def _create_columns(self, column_names, logical_types, semantic_types):
        """Create a dictionary with column names as keys and new DataColumn objects
            as values, while assigning any values that are passed for logical types or
            semantic types to the new column."""
        data_columns = {}
        for name in column_names:
            if logical_types and name in logical_types:
                logical_type = logical_types[name]
            else:
                logical_type = None
            if semantic_types and name in semantic_types:
                semantic_type = semantic_types[name]
            else:
                semantic_type = {}
            dc = DataColumn(self.dataframe[name], logical_type, semantic_type)
            data_columns[dc.name] = dc
        return data_columns

    @property
    def logical_types(self):
        return {dc.name: dc.logical_type for dc in self.columns.values()}

    @property
    def physical_types(self):
        return {dc.name: dc.dtype for dc in self.columns.values()}

    @property
    def semantic_types(self):
        return {dc.name: dc.semantic_types for dc in self.columns.values()}

    def _update_columns(self, new_columns):
        """Update the DataTable columns based on items contained in the
            provided new_columns dictionary"""
        for name, column in new_columns.items():
            self.columns[name] = column

    def set_logical_types(self, logical_types):
        """Update the logical type for any columns names in the provided logical_types
            dictionary, retaining any semantic types for the column. Replaces the existing
            column with a new column object."""
        _check_logical_types(self.dataframe, logical_types)
        # Get any existing semantic tags to retain on new columns
        semantic_types = {}
        for name in logical_types.keys():
            semantic_types[name] = self.columns[name].semantic_types
        cols_to_update = self._create_columns(logical_types.keys(),
                                              logical_types,
                                              semantic_types)
        self._update_columns(cols_to_update)

    def add_semantic_types(self, semantic_types):
        # semantic_types: (dict -> SemanticTag/str)
        # will not overwrite, will append to set
        pass

    def remove_semantic_types(self, semantic_types):
        # semantic_types: (dict -> SemanticTag/str)
        # remove tag from a data column
        pass

    def set_semantic_types(self, semantic_types):
        """Update the semantic types for any column names in the provided semantic_types
            dictionary. Replaces the existing semantic types with the new values."""
        _check_semantic_types(self.dataframe, semantic_types)
        for name in semantic_types.keys():
            self.columns[name].set_semantic_types(semantic_types[name])

    @property
    def df(self):
        return self.dataframe

    def to_pandas_dataframe(self):
        return self.dataframe


def _validate_params(dataframe, name, index, time_index, logical_types, semantic_types):
    """Check that values supplied during DataTable initialization are valid"""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError('Dataframe must be a pandas.DataFrame')
    _check_unique_column_names(dataframe)
    if name and not isinstance(name, str):
        raise TypeError('DataTable name must be a string')
    if index:
        _check_index(dataframe, index)
    if time_index:
        _check_time_index(dataframe, index)
    if logical_types:
        _check_logical_types(dataframe, logical_types)
    if semantic_types:
        _check_semantic_types(dataframe, semantic_types)


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


def _check_logical_types(dataframe, logical_types):
    if not isinstance(logical_types, dict):
        raise TypeError('logical_types must be a dictionary')
    cols_not_found = set(logical_types.keys()).difference(set(dataframe.columns))
    if cols_not_found:
        raise LookupError('logical_types contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')


def _check_semantic_types(dataframe, semantic_types):
    if not isinstance(semantic_types, dict):
        raise TypeError('semantic_types must be a dictionary')
    cols_not_found = set(semantic_types.keys()).difference(set(dataframe.columns))
    if cols_not_found:
        raise LookupError('semantic_types contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')
