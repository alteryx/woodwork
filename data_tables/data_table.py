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
            semantic_types ():
            logical_types (dict[str -> LogicalType], optional): Dictionary mapping column names in
                the dataframe to the LogicalType for the column. LogicalTypes will be inferred
                for any columns not present in the dictionary.
            copy_dataframe (bool, optional): If True, a copy of the input dataframe will be made
                prior to creating the DataTable. Otherwise, use a reference to the input dataframe.
        """
        # Check that inputs are valid
        _validate_params(dataframe, name, index, time_index)

        if copy_dataframe:
            self.dataframe = dataframe.copy()
        else:
            self.dataframe = dataframe

        self.name = name
        self.index = index
        self.time_index = time_index

        # Infer logical types and create columns
        self.columns = self._create_columns(self.dataframe, logical_types)

    def __repr__(self):
        # print out data column names, pandas dtypes, Logical Types & Semantic Tags
        # similar to df.types
        pass

    def _create_columns(self, dataframe, user_logical_types):
        data_columns = {}
        for col in self.dataframe.columns:
            if user_logical_types and col in user_logical_types:
                logical_type = user_logical_types[col]
            else:
                # if user not specifying Logical Type
                logical_type = infer_logical_type(self.dataframe[col])
            dc = DataColumn(self.dataframe[col], logical_type, set())
            data_columns[dc.name] = dc
        return data_columns

    def set_logical_types(self, logical_types):
        # logical_types: (dict -> LogicalType/str)
        # change the data column logical types
        # implementation detail --> create new data column, do not update
        pass

    def add_semantic_types(self, semantic_types):
        # semantic_types: (dict -> SemanticTag/str)
        # will not overwrite, will append to set
        pass

    def remove_semantic_types(self, semantic_types):
        # semantic_types: (dict -> SemanticTag/str)
        # remove tag from a data column
        pass

    def set_semantic_types(self, semantic_types):
        # semantic_types: (dict -> SemanticTag/str)
        # overwrite the tags
        pass

    @property
    def df(self):
        return self.dataframe

    def to_pandas_dataframe(self):
        return self.dataframe


def _validate_params(dataframe, name, index, time_index):
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


def infer_logical_type(series):
    # copy some of the logical from featuretools.infer_variable_types
    return
