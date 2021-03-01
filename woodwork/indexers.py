import copy

import pandas as pd

import woodwork as ww
from woodwork.accessor_utils import _is_dataframe, _is_series
from woodwork.utils import _new_dt_including, import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


class _iLocIndexer:
    def __init__(self, ww_data):
        self.ww_data = ww_data
        if isinstance(ww_data, ww.DataTable):
            self.underlying_data = ww_data._dataframe
            if dd and isinstance(self.underlying_data, dd.DataFrame):
                raise TypeError("iloc is not supported for Dask DataTables")
        elif isinstance(ww_data, ww.DataColumn):
            self.underlying_data = ww_data._series
            if dd and isinstance(self.underlying_data, dd.Series):
                raise TypeError("iloc is not supported for Dask DataColumns")

    def __getitem__(self, key):
        selection = self.underlying_data.iloc[key]
        if isinstance(selection, pd.Series) or (ks and isinstance(selection, ks.Series)):
            col_name = selection.name
            if isinstance(self.ww_data, ww.DataTable) and set(selection.index.values) == set(self.ww_data.columns):
                # return selection as series if series of one row.
                return selection
            if isinstance(self.ww_data, ww.DataTable):
                logical_type = self.ww_data.logical_types.get(col_name, None)
                semantic_tags = self.ww_data.semantic_tags.get(col_name, None)
            else:
                logical_type = self.ww_data.logical_type or None
                semantic_tags = self.ww_data.semantic_tags or None
            if semantic_tags is not None:
                semantic_tags = semantic_tags - {'index'} - {'time_index'}
            name = self.ww_data.name
            return ww.DataColumn(selection,
                                 logical_type=logical_type,
                                 semantic_tags=semantic_tags,
                                 use_standard_tags=self.ww_data.use_standard_tags,
                                 name=name)
        elif isinstance(selection, pd.DataFrame) or (ks and isinstance(selection, ks.DataFrame)):
            return _new_dt_including(self.ww_data, selection)
        else:
            # singular value
            return selection


class _iLocIndexerAccessor:
    def __init__(self, data):
        self.data = data
        if dd and isinstance(self.data, dd.DataFrame):
            raise TypeError("iloc is not supported for Dask DataFrames")
        elif dd and isinstance(data, dd.Series):
            raise TypeError("iloc is not supported for Dask Series")

    def __getitem__(self, key):
        selection = self.data.iloc[key]
        return _process_selection(selection, self.data)


class _locIndexerAccessor:
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
            schema['semantic_tags'] = schema['semantic_tags'] - {'index'} - {'time_index'}
        else:
            # Selecting a new Series from an existing Series
            schema = copy.deepcopy(original_data.ww._schema)
        if schema:
            del schema['dtype']
            selection.ww.init(**schema, use_standard_tags=original_data.ww.use_standard_tags)
    elif _is_dataframe(selection):
        # Selecting a new DataFrame from an existing DataFrame
        schema = original_data.ww.schema
        new_schema = schema._get_subset_schema(list(selection.columns))
        selection.ww.init(schema=new_schema)
    # Selecting a single value or return selection from above
    return selection
