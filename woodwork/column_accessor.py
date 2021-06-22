import copy
import warnings
import weakref

import pandas as pd

from woodwork.accessor_utils import (
    _is_dataframe,
    _is_series,
    init_series,
    is_schema_valid
)
from woodwork.column_schema import ColumnSchema
from woodwork.exceptions import (
    ParametersIgnoredWarning,
    TypingInfoMismatchWarning,
    WoodworkNotInitError
)
from woodwork.indexers import _iLocIndexer, _locIndexer
from woodwork.logical_types import LatLong, Ordinal
from woodwork.table_schema import TableSchema
from woodwork.utils import (
    _get_column_logical_type,
    _is_valid_latlong_series,
    import_or_none
)

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


class WoodworkColumnAccessor:
    def __init__(self, series):
        self._series_weakref = weakref.ref(series)
        self._schema = None

    def init(self, logical_type=None, semantic_tags=None,
             use_standard_tags=True, description=None, metadata=None,
             schema=None, validate=True):
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
            schema (Woodwork.ColumnSchema, optional): Typing information to use for the Series instead of performing inference.
                Any other arguments provided will be ignored. Note that any changes made to the schema object after
                initialization will propagate to the Series. Similarly, to avoid unintended typing information changes,
                the same schema object should not be shared between Series.
            validate (bool, optional): Whether parameter and data validation should occur. Defaults to True. Warning:
                Should be set to False only when parameters and data are known to be valid.
                Any errors resulting from skipping validation with invalid inputs may not be easily understood.
        """

        if schema is not None:
            if validate:
                _validate_schema(schema, self._series)

            extra_params = []
            if logical_type is not None:
                extra_params.append('logical_type')
            if semantic_tags is not None:
                extra_params.append('semantic_tags')
            if description is not None:
                extra_params.append('description')
            if metadata is not None:
                extra_params.append('metadata')
            if not use_standard_tags:
                extra_params.append('use_standard_tags')
            if extra_params:
                warnings.warn("A schema was provided and the following parameters were ignored: " + ", ".join(extra_params), ParametersIgnoredWarning)

            self._schema = schema
        else:
            logical_type = _get_column_logical_type(self._series, logical_type, self._series.name)

            if validate:
                self._validate_logical_type(logical_type)

            self._schema = ColumnSchema(logical_type=logical_type,
                                        semantic_tags=semantic_tags,
                                        use_standard_tags=use_standard_tags,
                                        description=description,
                                        metadata=metadata,
                                        validate=validate)

    @property
    def _series(self):
        return self._series_weakref()

    @property
    def schema(self):
        return copy.deepcopy(self._schema)

    @property
    def description(self):
        """The description of the series"""
        if self._schema is None:
            _raise_init_error()
        return self._schema.description

    @description.setter
    def description(self, description):
        if self._schema is None:
            _raise_init_error()
        self._schema.description = description

    @property
    def iloc(self):
        """
        Integer-location based indexing for selection by position.
        ``.iloc[]`` is primarily integer position based (from ``0`` to
        ``length-1`` of the axis), but may also be used with a boolean array.

        If the selection result is a Series, Woodwork typing information will
        be initialized for the returned Series.

        Allowed inputs are:
            An integer, e.g. ``5``.
            A list or array of integers, e.g. ``[4, 3, 0]``.
            A slice object with ints, e.g. ``1:7``.
            A boolean array.
            A ``callable`` function with one argument (the calling Series, DataFrame
            or Panel) and that returns valid output for indexing (one of the above).
            This is useful in method chains, when you don't have a reference to the
            calling object, but would like to base your selection on some value.
        """
        if self._schema is None:
            _raise_init_error()
        return _iLocIndexer(self._series)

    @property
    def loc(self):
        """
        Access a group of rows by label(s) or a boolean array.

        ``.loc[]`` is primarily label based, but may also be used with a
        boolean array.

        If the selection result is a Series, Woodwork typing information will
        be initialized for the returned Series.

        Allowed inputs are:
            A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
            interpreted as a *label* of the index, and **never** as an
            integer position along the index).
            A list or array of labels, e.g. ``['a', 'b', 'c']``.
            A slice object with labels, e.g. ``'a':'f'``.
            A boolean array of the same length as the axis being sliced,
            e.g. ``[True, False, True]``.
            An alignable boolean Series. The index of the key will be aligned before
            masking.
            An alignable Index. The Index of the returned selection will be the input.
            A ``callable`` function with one argument (the calling Series or
            DataFrame) and that returns valid output for indexing (one of the above)
        """
        if self._schema is None:
            _raise_init_error()
        return _locIndexer(self._series)

    @property
    def logical_type(self):
        """The logical type of the series"""
        if self._schema is None:
            _raise_init_error()
        return self._schema.logical_type

    @property
    def metadata(self):
        """The metadata of the series"""
        if self._schema is None:
            _raise_init_error()
        return self._schema.metadata

    @metadata.setter
    def metadata(self, metadata):
        if self._schema is None:
            _raise_init_error()
        self._schema.metadata = metadata

    @property
    def semantic_tags(self):
        """The semantic tags assigned to the series"""
        if self._schema is None:
            _raise_init_error()
        return self._schema.semantic_tags

    @property
    def use_standard_tags(self):
        if self._schema is None:
            _raise_init_error()
        return self._schema.use_standard_tags

    def __eq__(self, other, deep=True):
        if not self._schema.__eq__(other._schema, deep=deep):
            return False
        if self._series.name != other._series.name:
            return False
        if deep and isinstance(self._series, pd.Series):
            return self._series.equals(other._series)
        return True

    def __getattr__(self, attr):
        # If the method is present on the Accessor, uses that method.
        # If the method is present on Series, uses that method.
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
        """Forwards the requested attribute onto the series object. Intercepts return value,
        attempting to initialize Woodwork with the current schema when a new Series is returned.
        Confirms schema is still valid for the original Series."""
        series_attr = getattr(self._series, attr)

        if callable(series_attr):
            def wrapper(*args, **kwargs):
                # Make Series call and intercept the result
                result = series_attr(*args, **kwargs)

                # Try to initialize Woodwork with the existing schema
                if _is_series(result):
                    valid_dtype = self._schema.logical_type._get_valid_dtype(type(result))
                    if str(result.dtype) == valid_dtype:
                        result.ww.init(schema=self.schema, validate=False)
                    else:
                        invalid_schema_message = 'dtype mismatch between original dtype, ' \
                            f'{valid_dtype}, and returned dtype, {result.dtype}'
                        warning_message = TypingInfoMismatchWarning().get_warning_message(attr,
                                                                                          invalid_schema_message,
                                                                                          'Series')
                        warnings.warn(warning_message, TypingInfoMismatchWarning)
                elif _is_dataframe(result):
                    # Initialize Woodwork if a valid table schema can be created from the original column schema
                    col_schema = self.schema
                    table_schema = TableSchema(column_names=[self.name],
                                               logical_types={self.name: col_schema.logical_type},
                                               semantic_tags={self.name: col_schema.semantic_tags},
                                               column_metadata={self.name: col_schema.metadata},
                                               use_standard_tags={self.name: col_schema.use_standard_tags},
                                               column_descriptions={self.name: col_schema.description},
                                               validate=False)
                    if is_schema_valid(result, table_schema):
                        result.ww.init(schema=table_schema)
                # Always return the results of the Series operation whether or not Woodwork is initialized
                return result
            return wrapper
        # Directly return non-callable Series attributes
        return series_attr

    def _validate_logical_type(self, logical_type):
        """Validates that a logical type is consistent with the series dtype. Performs additional type
        specific validation, as required."""
        valid_dtype = logical_type._get_valid_dtype(type(self._series))
        if valid_dtype != str(self._series.dtype):
            raise ValueError(f"Cannot initialize Woodwork. Series dtype '{self._series.dtype}' is "
                             f"incompatible with {logical_type} dtype. Try converting series "
                             f"dtype to '{valid_dtype}' before initializing or use the "
                             "woodwork.init_series function to initialize.")

        if isinstance(logical_type, Ordinal):
            logical_type._validate_data(self._series)
        elif isinstance(logical_type, LatLong):
            if not _is_valid_latlong_series(self._series):
                raise ValueError("Cannot initialize Woodwork. Series does not contain properly formatted "
                                 "LatLong data. Try reformatting before initializing or use the "
                                 "woodwork.init_series function to initialize.")

    def add_semantic_tags(self, semantic_tags):
        """Add the specified semantic tags to the set of tags.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to add
        """
        if self._schema is None:
            _raise_init_error()
        self._schema._add_semantic_tags(semantic_tags,
                                        self._series.name)

    def remove_semantic_tags(self, semantic_tags):
        """Removes specified semantic tags from the current tags.

        Args:
            semantic_tags (str/list/set): Semantic tag(s) to remove.
        """
        if self._schema is None:
            _raise_init_error()
        self._schema._remove_semantic_tags(semantic_tags,
                                           self._series.name)

    def reset_semantic_tags(self):
        """Reset the semantic tags to the default values. The default values
        will be either an empty set or a set of the standard tags based on the
        column logical type, controlled by the use_standard_tags property.

        Args:
            None
        """
        if self._schema is None:
            _raise_init_error()
        self._schema._reset_semantic_tags()

    def set_logical_type(self, logical_type):
        """Update the logical type for the series, clearing any previously set semantic tags,
        and returning a new series with Woodwork initialied.

        Args:
            logical_type (LogicalType, str): The new logical type to set for the series.

        Returns:
            Series: A new series with the updated logical type.
        """
        if self._schema is None:
            _raise_init_error()
        # Create a new series without a schema to prevent new series from sharing a common
        # schema with current series
        new_series = self._series.copy()
        new_series._schema = None
        return init_series(new_series,
                           logical_type=logical_type,
                           semantic_tags=None,
                           use_standard_tags=self._schema.use_standard_tags,
                           description=self.description,
                           metadata=copy.deepcopy(self.metadata))

    def set_semantic_tags(self, semantic_tags):
        """Replace current semantic tags with new values. If `use_standard_tags` is set
        to True for the series, any standard tags associated with the LogicalType of the
        series will be added as well.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to set
        """
        if self._schema is None:
            _raise_init_error()
        self._schema._set_semantic_tags(semantic_tags)


def _validate_schema(schema, series):
    if not isinstance(schema, ColumnSchema):
        raise TypeError('Provided schema must be a Woodwork.ColumnSchema object.')

    valid_dtype = schema.logical_type._get_valid_dtype(type(series))
    if str(series.dtype) != valid_dtype:
        raise ValueError(f"dtype mismatch between Series dtype {series.dtype}, and {schema.logical_type} dtype, {valid_dtype}")


def _raise_init_error():
    raise WoodworkNotInitError("Woodwork not initialized for this Series. Initialize by calling Series.ww.init")


@pd.api.extensions.register_series_accessor('ww')
class PandasColumnAccessor(WoodworkColumnAccessor):
    pass


if dd:
    @dd.extensions.register_series_accessor('ww')
    class DaskColumnAccessor(WoodworkColumnAccessor):
        pass


if ks:
    from databricks.koalas.extensions import register_series_accessor

    @register_series_accessor('ww')
    class KoalasColumnAccessor(WoodworkColumnAccessor):
        pass
