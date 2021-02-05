import pandas as pd

from woodwork.utils import _get_column_logical_type
from woodwork.schema_column import _get_column_dict


@pd.api.extensions.register_series_accessor('ww')
class WoodworkSeriesAccessor:
    def __init__(self, series):
        self._series = series
        self._schema = None

    def init(self, name=None, logical_type=None, **kwargs):
        # validate params
        # confirm name matches series, or update to match passed in name
        # logic should be in DataColumn - throws ColumnNameMismatchWarning if it needs to be changed 
        logical_type = _get_column_logical_type(self._series, logical_type, name)
        
        # Need a way for this to happen in place or return a whole new series with accessor initialized
        # _update_column_dtype(series, logical_type, name, inplace=True)
        
        self._schema = _get_column_dict(name, logical_type, **kwargs)

    @property
    def logical_type(self):
        return self._schema['logical_type']

    @property
    def schema(self):
        return self._schema

    @property
    def semantic_tags(self):
        return self._schema['semantic_tags']
