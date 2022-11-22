import copy
import warnings
import weakref
from typing import Dict, Optional, Union

import pandas as pd
from pandas.api import types as pdtypes

from woodwork.accessor_utils import (
    _check_column_schema,
    _is_dataframe,
    _is_series,
    init_series,
)
from woodwork.column_schema import ColumnSchema
from woodwork.exceptions import (
    ParametersIgnoredWarning,
    TypeValidationError,
    TypingInfoMismatchWarning,
)
from woodwork.indexers import _iLocIndexer, _locIndexer
from woodwork.logical_types import _NULLABLE_PHYSICAL_TYPES, LatLong, Ordinal
from woodwork.statistics_utils import _get_box_plot_info_for_column
from woodwork.table_schema import TableSchema
from woodwork.utils import _get_column_logical_type, import_or_none

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


class WoodworkColumnAccessor:
    def __init__(self, series):
        self._series_weakref = weakref.ref(series)
        self._schema = None

    def init(
        self,
        logical_type=None,
        semantic_tags=None,
        use_standard_tags=True,
        description=None,
        origin=None,
        metadata=None,
        schema=None,
        validate=True,
    ):
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
            origin (str, optional): Optional text specifying origin of the column (i.e. "base" or "engineered").
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
                extra_params.append("logical_type")
            if semantic_tags is not None:
                extra_params.append("semantic_tags")
            if description is not None:
                extra_params.append("description")
            if origin is not None:
                extra_params.append("origin")
            if metadata is not None:
                extra_params.append("metadata")
            if not use_standard_tags:
                extra_params.append("use_standard_tags")
            if extra_params:
                warnings.warn(
                    "A schema was provided and the following parameters were ignored: "
                    + ", ".join(extra_params),
                    ParametersIgnoredWarning,
                )

            self._schema = schema
        else:
            logical_type = _get_column_logical_type(
                self._series,
                logical_type,
                self._series.name,
            )

            if validate:
                if isinstance(logical_type, (Ordinal, LatLong)):
                    logical_type.validate(self._series)
                else:
                    valid_dtype = logical_type._get_valid_dtype(type(self._series))
                    if valid_dtype != str(self._series.dtype) and not (
                        pdtypes.is_integer_dtype(valid_dtype)
                        and pdtypes.is_float_dtype(self._series.dtype)
                    ):
                        raise TypeValidationError(
                            f"Cannot initialize Woodwork. Series dtype '{self._series.dtype}' is "
                            f"incompatible with {logical_type} LogicalType. Try converting series "
                            f"dtype to '{valid_dtype}' before initializing or use the "
                            "woodwork.init_series function to initialize.",
                        )

            self._schema = ColumnSchema(
                logical_type=logical_type,
                semantic_tags=semantic_tags,
                use_standard_tags=use_standard_tags,
                description=description,
                origin=origin,
                metadata=metadata,
                validate=validate,
            )

    @property
    def _series(self):
        return self._series_weakref()

    @property
    def schema(self):
        return copy.deepcopy(self._schema)

    @property
    @_check_column_schema
    def nullable(self):
        """Whether the column can contain null values."""
        dtype = self._schema.logical_type._get_valid_dtype(type(self._series))
        return dtype in _NULLABLE_PHYSICAL_TYPES

    @property
    @_check_column_schema
    def description(self):
        """The description of the series"""
        return self._schema.description

    @description.setter
    @_check_column_schema
    def description(self, description):
        self._schema.description = description

    @property
    @_check_column_schema
    def origin(self):
        """The origin of the series"""
        return self._schema.origin

    @origin.setter
    @_check_column_schema
    def origin(self, origin):
        self._schema.origin = origin

    @property
    @_check_column_schema
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
        return _iLocIndexer(self._series)

    @property
    @_check_column_schema
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
        return _locIndexer(self._series)

    @property
    @_check_column_schema
    def logical_type(self):
        """The logical type of the series"""
        return self._schema.logical_type

    @property
    @_check_column_schema
    def metadata(self):
        """The metadata of the series"""
        return self._schema.metadata

    @metadata.setter
    @_check_column_schema
    def metadata(self, metadata):
        self._schema.metadata = metadata

    @property
    @_check_column_schema
    def semantic_tags(self):
        """The semantic tags assigned to the series"""
        return self._schema.semantic_tags

    @property
    @_check_column_schema
    def use_standard_tags(self):
        return self._schema.use_standard_tags

    def __eq__(self, other, deep=True):
        if not self._schema.__eq__(other._schema, deep=deep):
            return False
        if self._series.name != other._series.name:
            return False
        if deep and isinstance(self._series, pd.Series):
            return self._series.equals(other._series)
        return True

    @_check_column_schema
    def __getattr__(self, attr):
        # Called if method is not present on the Accessor
        # If the method is present on Series, uses that method.
        if hasattr(self._series, attr):
            return self._make_series_call(attr)
        else:
            raise AttributeError(f"Woodwork has no attribute '{attr}'")

    @_check_column_schema
    def __repr__(self):
        msg = "<Series: {} ".format(self._series.name)
        msg += "(Physical Type = {}) ".format(self._series.dtype)
        msg += "(Logical Type = {}) ".format(self.logical_type)
        msg += "(Semantic Tags = {})>".format(self.semantic_tags)
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
                    valid_dtype = self._schema.logical_type._get_valid_dtype(
                        type(result),
                    )
                    if str(result.dtype) == valid_dtype:
                        result.ww.init(schema=self.schema, validate=False)
                    else:
                        invalid_schema_message = (
                            "dtype mismatch between original dtype, "
                            f"{valid_dtype}, and returned dtype, {result.dtype}"
                        )
                        warning_message = (
                            TypingInfoMismatchWarning().get_warning_message(
                                attr,
                                invalid_schema_message,
                                "Series",
                            )
                        )
                        warnings.warn(warning_message, TypingInfoMismatchWarning)
                elif _is_dataframe(result):
                    # Initialize Woodwork with a partial schema
                    col_schema = self.schema
                    col_name = self.name or result.columns.to_list()[0]
                    table_schema = TableSchema(
                        column_names=[col_name],
                        logical_types={col_name: col_schema.logical_type},
                        semantic_tags={col_name: col_schema.semantic_tags},
                        column_metadata={col_name: col_schema.metadata},
                        use_standard_tags={col_name: col_schema.use_standard_tags},
                        column_descriptions={col_name: col_schema.description},
                        column_origins={col_name: col_schema.origin},
                        validate=False,
                    )
                    result.ww.init_with_partial_schema(table_schema)
                # Always return the results of the Series operation whether or not Woodwork is initialized
                return result

            return wrapper
        # Directly return non-callable Series attributes
        return series_attr

    @_check_column_schema
    def add_semantic_tags(self, semantic_tags):
        """Add the specified semantic tags to the set of tags.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to add
        """
        self._schema._add_semantic_tags(semantic_tags, self._series.name)

    @_check_column_schema
    def remove_semantic_tags(self, semantic_tags):
        """Removes specified semantic tags from the current tags.

        Args:
            semantic_tags (str/list/set): Semantic tag(s) to remove.
        """
        self._schema._remove_semantic_tags(semantic_tags, self._series.name)

    @_check_column_schema
    def reset_semantic_tags(self):
        """Reset the semantic tags to the default values. The default values
        will be either an empty set or a set of the standard tags based on the
        column logical type, controlled by the use_standard_tags property.

        Args:
            None
        """
        self._schema._reset_semantic_tags()

    @_check_column_schema
    def set_logical_type(self, logical_type):
        """Update the logical type for the series, clearing any previously set semantic tags,
        and returning a new series with Woodwork initialied.

        Args:
            logical_type (LogicalType, str): The new logical type to set for the series.

        Returns:
            Series: A new series with the updated logical type.
        """
        # Create a new series without a schema to prevent new series from sharing a common
        # schema with current series
        new_series = self._series.copy()
        new_series._schema = None
        return init_series(
            new_series,
            logical_type=logical_type,
            semantic_tags=None,
            use_standard_tags=self._schema.use_standard_tags,
            description=self.description,
            origin=self.origin,
            metadata=copy.deepcopy(self.metadata),
        )

    @_check_column_schema
    def set_semantic_tags(self, semantic_tags):
        """Replace current semantic tags with new values. If `use_standard_tags` is set
        to True for the series, any standard tags associated with the LogicalType of the
        series will be added as well.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to set
        """
        self._schema._set_semantic_tags(semantic_tags)

    @_check_column_schema
    def get_outliers(
        self,
        method="best",
        quantiles: Optional[Dict[float, Union[int, float]]] = None,
        include_indices_and_values: bool = True,
        ignore_zeros: bool = False,
    ):
        """Gets the information necessary to create a box and whisker plot with outliers for a numeric column
        using the selected method.

        Args:
            method (str): The method to use when calculating the box and whiskers plot. Options are 'best', 'box_plot' and 'medcouple'.
                Defaults to 'best' at which point a heuristic will determine the appropriate method to use.
            quantiles (dict[float -> int or float], optional): A dictionary containing the quantiles for the data
                where the key indicates the quantile, and the value is the quantile's value for the data. If
                no quantiles are provided, they will be computed from the data.
            include_indices_and_values (bool, optional): Whether or not the lists containing individual
                outlier values and their indices will be included in the returned dictionary.
                Defaults to True.
            ignore_zeros (bool): Whether to ignore 0 values (not NaN values) when calculating the box plot and outliers.
                Defaults to False.

        Returns:
            (dict[str -> float,list[number]]): Returns a dictionary containing box plot information for the Series.
                The following elements will be found in the dictionary:

                - low_bound (float): the lowest data point in the dataset excluding any outliers - to be used as a whisker
                - high_bound (float): the highest point in the dataset excluding any outliers - to be used as a whisker
                - quantiles (list[float]): the quantiles used to determine the bounds.
                    If quantiles were passed in, will contain all quantiles passed in. Otherwise, contains the five
                    quantiles {0.0, 0.25, 0.5, 0.75, 1.0}.
                - low_values (list[float, int], optional): the values of the lower outliers.
                    Will not be included if ``include_indices_and_values`` is False.
                - high_values (list[float, int], optional): the values of the upper outliers
                    Will not be included if ``include_indices_and_values`` is False.
                - low_indices (list[int], optional): the corresponding index values for each of the lower outliers
                    Will not be included if ``include_indices_and_values`` is False.
                - high_indices (list[int], optional): the corresponding index values for each of the upper outliers
                    Will not be included if ``include_indices_and_values`` is False.
                - method (str): the method used to identify outliers
                - medcouple_stat (float): the medcouple statistic will be added in the response if the method selected
                    is medcouple
        """
        return _get_box_plot_info_for_column(
            self._series,
            method=method,
            quantiles=quantiles,
            include_indices_and_values=include_indices_and_values,
            ignore_zeros=ignore_zeros,
        )

    @_check_column_schema
    def box_plot_dict(
        self,
        quantiles: Optional[Dict[int, int]] = None,
        include_indices_and_values: bool = True,
        ignore_zeros: bool = False,
    ):
        """Gets the information necessary to create a box and whisker plot with outliers for a numeric column
        using the IQR method.

        Args:
            quantiles (dict[float -> float], optional): A dictionary containing the quantiles for the data
                where the key indicates the quantile, and the value is the quantile's value for the data. If
                no quantiles are provided, they will be computed from the data.
            include_indices_and_values (bool, optional): Whether or not the lists containing individual
                outlier values and their indices will be included in the returned dictionary.
                Defaults to True.
            ignore_zeros (bool): Whether to ignore 0 values (not NaN values) when calculating the box plot and outliers.
                Defaults to False.

        Note:
            The minimum quantiles necessary for building a box plot using the IQR method are the
            minimum value (0.0 in the quantiles dict), first quartile (0.25), third quartile (0.75), and maximum value (1.0).
            If no quantiles are provided, the following quantiles will be calculated:
            {0.0, 0.25, 0.5, 0.75, 1.0}, which correspond to {min, first quantile, median, third quantile, max}.

        Returns:
            (dict[str -> float,list[number]]): Returns a dictionary containing box plot information for the Series.
                The following elements will be found in the dictionary:

                - low_bound (float): the lowest data point in the dataset excluding any outliers - to be used as a whisker
                - high_bound (float): the highest point in the dataset excluding any outliers - to be used as a whisker
                - quantiles (list[float]): the quantiles used to determine the bounds.
                    If quantiles were passed in, will contain all quantiles passed in. Otherwise, contains the five
                    quantiles {0.0, 0.25, 0.5, 0.75, 1.0}.
                - low_values (list[float, int], optional): the values of the lower outliers.
                    Will not be included if ``include_indices_and_values`` is False.
                - high_values (list[float, int], optional): the values of the upper outliers
                    Will not be included if ``include_indices_and_values`` is False.
                - low_indices (list[int], optional): the corresponding index values for each of the lower outliers
                    Will not be included if ``include_indices_and_values`` is False.
                - high_indices (list[int], optional): the corresponding index values for each of the upper outliers
                    Will not be included if ``include_indices_and_values`` is False.
                - method (str): the method used to identify outliers, in this case box_plot
        """
        return _get_box_plot_info_for_column(
            self._series,
            method="box_plot",
            quantiles=quantiles,
            include_indices_and_values=include_indices_and_values,
            ignore_zeros=ignore_zeros,
        )

    @_check_column_schema
    def medcouple_dict(
        self,
        quantiles: Optional[Dict[int, int]] = None,
        include_indices_and_values: bool = True,
        ignore_zeros: bool = False,
    ):
        """Gets the information necessary to create a box and whisker plot with outliers for a numeric column
        using the Medcouple statistic method.

        Args:
            quantiles (dict[float -> float], optional): A dictionary containing the quantiles for the data
                where the key indicates the quantile, and the value is the quantile's value for the data. If
                no quantiles are provided, they will be computed from the data.
            include_indices_and_values (bool, optional): Whether or not the lists containing individual
                outlier values and their indices will be included in the returned dictionary.
                Defaults to True.
            ignore_zeros (bool): Whether to ignore 0 values (not NaN values) when calculating the box plot and outliers.
                Defaults to False.

        Note:
            The minimum quantiles necessary for building a box plot using the Medcouple statistic method are the
            minimum value (0.0 in the quantiles dict), first quartile (0.25), third quartile (0.75), and maximum value (1.0).
            If no quantiles are provided, the following quantiles will be calculated:
            {0.0, 0.25, 0.5, 0.75, 1.0}, which correspond to {min, first quantile, median, third quantile, max}.

        Returns:
            (dict[str -> float,list[number]]): Returns a dictionary containing box plot information for the Series.
                The following elements will be found in the dictionary:

                - low_bound (float): the lowest data point in the dataset excluding any outliers - to be used as a whisker
                - high_bound (float): the highest point in the dataset excluding any outliers - to be used as a whisker
                - quantiles (list[float]): the quantiles used to determine the bounds.
                    If quantiles were passed in, will contain all quantiles passed in. Otherwise, contains the five
                    quantiles {0.0, 0.25, 0.5, 0.75, 1.0}.
                - low_values (list[float, int], optional): the values of the lower outliers.
                    Will not be included if ``include_indices_and_values`` is False.
                - high_values (list[float, int], optional): the values of the upper outliers
                    Will not be included if ``include_indices_and_values`` is False.
                - low_indices (list[int], optional): the corresponding index values for each of the lower outliers
                    Will not be included if ``include_indices_and_values`` is False.
                - high_indices (list[int], optional): the corresponding index values for each of the upper outliers
                    Will not be included if ``include_indices_and_values`` is False.
                - method (str): the method used to identify outliers, in this case medcouple.
                - medcouple_stat (float): the medcouple statistic
        """
        return _get_box_plot_info_for_column(
            self._series,
            method="medcouple",
            quantiles=quantiles,
            include_indices_and_values=include_indices_and_values,
            ignore_zeros=ignore_zeros,
        )

    @_check_column_schema
    def validate_logical_type(self, return_invalid_values=False):
        """Validates series data based on the logical type.
        If a column's dtype does not match its logical type's required dtype,
        will raise a TypeValidationError even when return_invalid_indices is True.

        Args:
            return_invalid_values (bool): Whether or not to return invalid data values

        Returns:
            Series: If return_invalid_values is True, returns invalid data values.
        """
        return self.logical_type.validate(
            series=self._series,
            return_invalid_values=return_invalid_values,
        )


def _validate_schema(schema, series):
    if not isinstance(schema, ColumnSchema):
        raise TypeError("Provided schema must be a Woodwork.ColumnSchema object.")

    valid_dtype = schema.logical_type._get_valid_dtype(type(series))
    if str(series.dtype) != valid_dtype:
        raise ValueError(
            f"dtype mismatch between Series dtype {series.dtype}, and {schema.logical_type} dtype, {valid_dtype}",
        )


@pd.api.extensions.register_series_accessor("ww")
class PandasColumnAccessor(WoodworkColumnAccessor):
    pass


if dd:

    @dd.extensions.register_series_accessor("ww")
    class DaskColumnAccessor(WoodworkColumnAccessor):
        pass


if ps:
    from pyspark.pandas.extensions import register_series_accessor

    @register_series_accessor("ww")
    class SparkColumnAccessor(WoodworkColumnAccessor):
        pass
