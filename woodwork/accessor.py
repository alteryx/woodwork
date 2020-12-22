import woodwork as ww
import dask.dataframe as dd
from databricks.koalas.extensions import register_dataframe_accessor, register_series_accessor
import pandas as pd


# DataFrame Accessors
@pd.api.extensions.register_dataframe_accessor('ww')
class DataTableAccessor:
    def __init__(self, dataframe):
        self._datatable = ww.DataTable(dataframe)

    def __getattribute__(self, attr):
        return _get_attr_from_child(self, '_datatable', attr)

    def init(self, **kwargs):
        '''Initialize a DataTable the pandas accessor with any relevant DataTable arguments.
        '''
        self._datatable = ww.DataTable(self._datatable.df, **kwargs)

    @property
    def dt(self):
        return self._datatable

    @dt.setter
    def dt(self, datatable):
        '''To be used any time a DataTable method returns a new DataTable object
        '''
        self._datatable = datatable


@dd.extensions.register_dataframe_accessor('ww')
class DaskDataTableAccessor(DataTableAccessor):
    pass


@register_dataframe_accessor('ww')
class KoalasDataTableAccessor(DataTableAccessor):
    pass

# DataColumn Accessors
@pd.api.extensions.register_series_accessor('ww')
class DataColumnAccessor:
    def __init__(self, series):
        self._datacolumn = ww.DataColumn(series)

    def __getattribute__(self, attr):
        return _get_attr_from_child(self, '_datacolumn', attr)

    def init(self, **kwargs):
        '''Initialize a DataColumn the pandas accessor with any relevant DataColumn arguments.
        '''
        self._datacolumn = ww.DataColumn(self._datacolumn.to_series(), **kwargs)

    @property
    def dc(self):
        return self._datacolumn

    @dc.setter
    def dc(self, datacolumn):
        '''To be used any time a DataColumn method returns a new DataColumn object
        '''
        self._datacolumn = datacolumn


@dd.extensions.register_series_accessor('ww')
class DaskDataColumnAccessor(DataColumnAccessor):
    pass


@register_series_accessor('ww')
class KoalasDataColumnAccessor(DataColumnAccessor):
    pass


def _get_attr_from_child(obj, child, attr):
    child = object.__getattribute__(obj, child)

    if hasattr(child, attr):
        child_attr = getattr(child, attr)

        # logical_type attr can return an uninstantiated LogicalType which we don't want to interpret as callable
        if callable(child_attr) and attr != 'logical_type':
            def wrapper(*args, **kwargs):
                return child_attr(*args, **kwargs)
            return wrapper
        else:
            return child_attr
    return object.__getattribute__(obj, attr)
