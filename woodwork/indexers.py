import copy

from woodwork.accessor_utils import _is_dataframe, _is_series
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


class _iLocIndexer:
    def __init__(self, data):
        self.data = data
        if dd and isinstance(self.data, dd.DataFrame):
            raise TypeError("iloc is not supported for Dask DataFrames")
        elif dd and isinstance(data, dd.Series):
            raise TypeError("iloc is not supported for Dask Series")

    def __getitem__(self, key):
        selection = self.data.iloc[key]
        return _process_selection(selection, self.data)


class _locIndexer:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        selection = self.data.loc[key]
        return _process_selection(selection, self.data)


def _process_selection(selection, original_data):
    if _is_series(selection):
        if dd and isinstance(selection, dd.Series):
            # Dask index values are a delayed object - can't compare below without computing
            index_vals = selection.index.values.compute()
        else:
            index_vals = selection.index.values
        if _is_dataframe(original_data) and set(index_vals) == set(original_data.columns):
            # Selecting a single row from a DataFrame, returned as Series without Woodwork initialized
            schema = None
        elif _is_dataframe(original_data):
            # Selecting a single column from a DataFrame
            schema = original_data.ww.schema.columns[selection.name]
        else:
            # Selecting a new Series from an existing Series
            schema = original_data.ww._schema
        if schema:
            selection.ww.init(schema=copy.deepcopy(schema), validate=False)
    elif _is_dataframe(selection):
        # Selecting a new DataFrame from an existing DataFrame
        schema = original_data.ww.schema
        new_schema = schema._get_subset_schema(list(selection.columns))
        selection.ww.init(schema=new_schema, validate=False)
    # Selecting a single value or return selection from above
    return selection
