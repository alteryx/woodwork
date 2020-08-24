from data_tables.data_column import DataColumn


class DataTable(object):
    def __init__(self, dataframe, name,
                 index=None,
                 time_index=None,
                 semantic_types=None,
                 logical_types=None,
                 copy_dataframe=False):

        # Check unique colum names
        check_unique_column_names(dataframe)
        # Check index column is unique
        if index:
            check_index(dataframe, index)

        # infer physical types via pandas, ensure df not mutated
        self.dataframe = self.dataframe.infer_types()

        # infer logical types and create columns
        self.columns = self.create_columns(self.dataframe)
        self.name = name  # optional
        self.index = index  # optional, name of the data column
        self.time_index = time_index  # optional, name of the data column
        self.df._infer_types()  # sets columns

    def __repr__(self):
        # print out data column names, pandas dtypes, Logical Types & Semantic Tags
        # similar to df.types
        pass

    def create_columns(self, dataframe, user_logical_types):
        data_columns = {}
        for col in self.dataframe.columns:
            if col not in user_logical_types:
                # if user not specifying Logical Type
                logical_type = infer_logical_type(self.dataframe[col])
            else:
                logical_type = user_logical_types[col]
            dc = DataColumn(self.dataframe[col], logical_type)
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


def infer_logical_type(series):
    # copy some of the logical from featuretools.infer_variable_types
    return


def check_unique_column_names(datatable):
    pass


def check_index(datatable):
    pass
