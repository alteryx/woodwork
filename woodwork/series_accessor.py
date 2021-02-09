import warnings

import pandas as pd

from woodwork.exceptions import ColumnNameMismatchWarning
from woodwork.schema_column import (
    _get_column_dict,
    _validate_description,
    _validate_metadata
)
from woodwork.utils import _get_column_logical_type


@pd.api.extensions.register_series_accessor('ww')
class WoodworkSeriesAccessor:
    def __init__(self, series):
        self._series = series
        self._schema = None

    def init(self, name=None, logical_type=None, semantic_tags=None,
             use_standard_tags=True, description=None, metadata=None):
        self._set_name(name)

        logical_type = _get_column_logical_type(self._series, logical_type, self.name)
        if logical_type.pandas_dtype != str(self._series.dtype):
            raise ValueError(f"Cannot initialize Woodwork. Series dtype is incompatible with {logical_type} dtype. " \
                f"Try converting series dtype to {logical_type.pandas_dtype} before initializing.")

        self._schema = _get_column_dict(name=self.name,
                                        logical_type=logical_type,
                                        semantic_tags=semantic_tags,
                                        use_standard_tags=use_standard_tags,
                                        description=description,
                                        metadata=metadata)

    @property
    def description(self):
        return self._schema['description']

    @description.setter
    def description(self, description):
        _validate_description(description)
        self._schema['description'] = description

    @property
    def logical_type(self):
        return self._schema['logical_type']

    @property
    def metadata(self):
        return self._schema['metadata']

    @metadata.setter
    def metadata(self, metadata):
        _validate_metadata(metadata)
        self._schema['metadata'] = metadata

    @property
    def name(self):
        return self._series.name

    @property
    def schema(self):
        return self._schema

    @property
    def semantic_tags(self):
        return self._schema['semantic_tags']

    def _set_name(self, name=None):
        if name is not None and self._series.name is not None and name != self._series.name:
            warnings.warn(ColumnNameMismatchWarning().get_warning_message(self._series.name, name),
                          ColumnNameMismatchWarning)
        if name is not None:
            self._series.name = name
