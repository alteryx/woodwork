import copy
import warnings

import pandas as pd

from woodwork.accessor_utils import init_series
from woodwork.exceptions import TypingInfoMismatchWarning
from woodwork.logical_types import Ordinal
from woodwork.schema_column import (
    _add_semantic_tags,
    _get_column_dict,
    _remove_semantic_tags,
    _reset_semantic_tags,
    _set_semantic_tags,
    _validate_description,
    _validate_metadata
)
from woodwork.utils import _get_column_logical_type


@pd.api.extensions.register_series_accessor('ww')
class WoodworkColumnAccessor:
    def __init__(self, series):
        self._series = series
        self._schema = None

    def init(self, logical_type=None, semantic_tags=None,
             use_standard_tags=True, description=None, metadata=None):
        """Initializes Woodwork typing information for a Series.

        Args:
            logical_type (LogicalType or str, optional): The logical type that should be assigned
                to the series. If no value is provided, the LogicalType for the series will
                be inferred. If the LogicalType provided or inferred does not have a dtype that
                is compatible with the series dtype, an error will be raised.
            semantic_tags (str or list or set, optional): Semantic tags to assign to the series.
                Defaults to an empty set if not specified. There are two options for
                specifying the semantic tags:
                (str) If only one semantic tag is being set, a single string can be passed.
                (list or set) If multiple tags are being set, a list or set of strings can be passed.
            use_standard_tags (bool, optional): If True, will add standard semantic tags to the series
                based on the inferred or specified logical type of the series. Defaults to True.
            description (str, optional): Optional text describing the contents of the series.
            metadata (dict[str -> json serializable], optional): Metadata associated with the series.
        """
        logical_type = _get_column_logical_type(self._series, logical_type, self._series.name)

        self._validate_logical_type(logical_type)
        self.use_standard_tags = use_standard_tags

        self._schema = _get_column_dict(name=self._series.name,
                                        logical_type=logical_type,
                                        semantic_tags=semantic_tags,
                                        use_standard_tags=self.use_standard_tags,
                                        description=description,
                                        metadata=metadata)

    @property
    def description(self):
        """The description of the series"""
        if self._schema is None:
            _raise_init_error()
        return self._schema['description']

    @description.setter
    def description(self, description):
        _validate_description(description)
        self._schema['description'] = description

    @property
    def logical_type(self):
        """The logical type of the series"""
        if self._schema is None:
            _raise_init_error()
        return self._schema['logical_type']

    @property
    def metadata(self):
        """The metadata of the series"""
        if self._schema is None:
            _raise_init_error()
        return self._schema['metadata']

    @metadata.setter
    def metadata(self, metadata):
        _validate_metadata(metadata)
        self._schema['metadata'] = metadata

    @property
    def semantic_tags(self):
        """The semantic tags assigned to the series"""
        if self._schema is None:
            _raise_init_error()
        return self._schema['semantic_tags']

    def __eq__(self, other):
        if self._schema != other._schema:
            return False
        return self._series.equals(other._series)

    def __getattr__(self, attr):
        '''
            If the method is present on the Accessor, uses that method.
            If the method is present on Series, uses that method.
        '''
        if self._schema is None:
            _raise_init_error()
        if hasattr(self._series, attr):
            return self._make_series_call(attr)
        else:
            raise AttributeError(f"Woodwork has no attribute '{attr}'")

    def __repr__(self):
        if self._schema is None:
            _raise_init_error()
        msg = u"<Series: {} ".format(self._series.name)
        msg += u"(Physical Type = {}) ".format(self._series.dtype)
        msg += u"(Logical Type = {}) ".format(self.logical_type)
        msg += u"(Semantic Tags = {})>".format(self.semantic_tags)
        return msg

    def _make_series_call(self, attr):
        '''
        Forwards the requested attribute onto the series object.
        Intercepts return value, attempting to initialize Woodwork with the current schema
        when a new Series is returned.
        Confirms schema is still valid for the original Series.
        '''
        series_attr = getattr(self._series, attr)

        if callable(series_attr):
            def wrapper(*args, **kwargs):
                # Make Series call and intercept the result
                result = series_attr(*args, **kwargs)

                # Try to initialize Woodwork with the existing Schema
                if isinstance(result, pd.Series):
                    if result.dtype == self._schema['logical_type'].pandas_dtype:
                        schema = copy.deepcopy(self._schema)
                        # We don't need to pass dtype from the schema to init
                        del schema['dtype']
                        result.ww.init(**schema)
                    else:
                        invalid_schema_message = 'dtype mismatch between original dtype, ' \
                            f'{self._schema["logical_type"].pandas_dtype}, and returned dtype, {result.dtype}'
                        warning_message = TypingInfoMismatchWarning().get_warning_message(attr,
                                                                                          invalid_schema_message,
                                                                                          'Series')
                        warnings.warn(warning_message, TypingInfoMismatchWarning)
                # Always return the results of the Series operation whether or not Woodwork is initialized
                return result
            return wrapper
        # Directly return non-callable Series attributes
        return series_attr

    def _validate_logical_type(self, logical_type):
        """Validates that a logical type is consistent with the series dtype. Performs additional type
        specific validation, as required."""
        if logical_type.pandas_dtype != str(self._series.dtype):
            raise ValueError(f"Cannot initialize Woodwork. Series dtype '{self._series.dtype}' is "
                             f"incompatible with {logical_type} dtype. Try converting series "
                             f"dtype to '{logical_type.pandas_dtype}' before initializing or use the "
                             "woodwork.init_series function to initialize.")

        if isinstance(logical_type, Ordinal):
            logical_type._validate_data(self._series)

    def add_semantic_tags(self, semantic_tags):
        """Add the specified semantic tags to the set of tags.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to add
        """
        self._schema['semantic_tags'] = _add_semantic_tags(semantic_tags,
                                                           self.semantic_tags,
                                                           self._series.name)

    def remove_semantic_tags(self, semantic_tags):
        """Removes specified semantic tags from the current tags.

        Args:
            semantic_tags (str/list/set): Semantic tag(s) to remove.
        """
        self._schema['semantic_tags'] = _remove_semantic_tags(semantic_tags,
                                                              self.semantic_tags,
                                                              self._series.name,
                                                              self.logical_type.standard_tags,
                                                              self.use_standard_tags)

    def reset_semantic_tags(self):
        """Reset the semantic tags to the default values. The default values
        will be either an empty set or a set of the standard tags based on the
        column logical type, controlled by the use_standard_tags property.
        """
        self._schema['semantic_tags'] = _reset_semantic_tags(self.logical_type.standard_tags,
                                                             self.use_standard_tags)

    def set_logical_type(self, logical_type):
        """Update the logical type for the series, clearing any previously set semantic tags,
        and returning a new Series.

        Args:
            logical_type (LogicalType, str): The new logical type to set for the series.

        Returns:
            Series: A new series with the updated logical type.
        """
        # Create a new series without a schema to prevent new series from sharing a common
        # schema with current series
        new_series = self._series.copy()
        new_series._schema = None
        return init_series(new_series,
                           logical_type=logical_type,
                           semantic_tags=None,
                           use_standard_tags=self.use_standard_tags,
                           description=self.description,
                           metadata=copy.deepcopy(self.metadata))

    def set_semantic_tags(self, semantic_tags):
        """Replace current semantic tags with new values. If `use_standard_tags` is set
        to True for the series, any standard tags associated with the LogicalType of the
        series will be added as well.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to set
        """
        self._schema['semantic_tags'] = _set_semantic_tags(semantic_tags,
                                                           self.logical_type.standard_tags,
                                                           self.use_standard_tags)


def _raise_init_error():
    raise AttributeError("Woodwork not initialized for this Series. Initialize by calling Series.ww.init")
