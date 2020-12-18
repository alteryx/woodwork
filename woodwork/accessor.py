import woodwork as ww
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("ww")
class DataTableAccessor:
    def __init__(self, pandas_obj):
        self._datatable = ww.DataTable(pandas_obj)

    def __getattribute__(self, attr):
        dt = object.__getattribute__(self, '_datatable')

        if hasattr(dt, attr):
            dt_attr = getattr(dt, attr)
            if callable(dt_attr):
                def wrapper(*args, **kwargs):
                    return dt_attr(*args, **kwargs)
                return wrapper
            else:
                return dt_attr
        return object.__getattribute__(self, attr)

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
