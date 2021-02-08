import warnings

import pandas as pd

from woodwork.exceptions import ColumnNameMismatchWarning
from woodwork.schema_column import _get_column_dict
from woodwork.utils import _get_column_logical_type


@pd.api.extensions.register_series_accessor('ww')
class WoodworkSeriesAccessor:
    def __init__(self, series):
        self._series = series
        self._schema = None

    def init(self, name=None, logical_type=None, **kwargs):
        # validate params
        self._set_name(name)

        # logic should be in DataColumn - throws ColumnNameMismatchWarning if it needs to be changed
        logical_type = _get_column_logical_type(self._series, logical_type, name)

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

    @property
    def name(self):
        return self._series.name

    def _set_name(self, name=None):
        if name is not None and self._series.name is not None and name != self._series.name:
            warnings.warn(ColumnNameMismatchWarning().get_warning_message(self._series.name, name),
                          ColumnNameMismatchWarning)
        if name is not None:
            self._series.name = name
