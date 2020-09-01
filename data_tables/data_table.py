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
            copy_dataframe (bool, optional):
        """
        _validate_params(dataframe, name)
        self.dataframe = dataframe

        # Check unique colum names
        check_unique_column_names(dataframe)
        # Check index column is unique
        if index:
            check_index(dataframe, index)

        # infer logical types and create columns
        self.columns = self.create_columns(self.dataframe, logical_types)
        self.name = name
        self.index = index
        self.time_index = time_index

    def __repr__(self):
        # print out data column names, pandas dtypes, Logical Types & Semantic Tags
        # similar to df.types
        pass

    def create_columns(self, dataframe, user_logical_types):
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
        return self.dataframe.copy()

    def to_pandas_dataframe(self):
        return self.dataframe.copy()


def _validate_params(dataframe, name):
    assert isinstance(dataframe, pd.DataFrame), 'Dataframe must be a pandas.DataFrame'
    if name:
        assert isinstance(name, str), 'DataTable name must be a string'


def infer_logical_type(series):
    # copy some of the logical from featuretools.infer_variable_types
    return


def check_unique_column_names(datatable):
    pass


def check_index(datatable, index):
    pass
