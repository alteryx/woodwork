import copy
import warnings
import weakref
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

import pandas as pd

from woodwork.accessor_utils import (
    _check_table_schema,
    _is_dask_dataframe,
    _is_dataframe,
    _is_spark_dataframe,
    get_invalid_schema_message,
    init_series,
)
from woodwork.exceptions import (
    ColumnBothIgnoredAndSetError,
    ColumnNotPresentError,
    IndexTagRemovedWarning,
    ParametersIgnoredWarning,
    TypingInfoMismatchWarning,
    WoodworkNotInitError,
)
from woodwork.indexers import _iLocIndexer, _locIndexer
from woodwork.logical_types import Datetime, LogicalType
from woodwork.serializers.serializer_base import typing_info_to_dict
from woodwork.serializers.utils import get_serializer
from woodwork.statistics_utils import (
    _get_dependence_dict,
    _get_describe_dict,
    _get_valid_mi_columns,
    _get_value_counts,
    _infer_temporal_frequencies,
)
from woodwork.table_schema import TableSchema
from woodwork.type_sys.utils import _is_numeric_series, col_is_datetime
from woodwork.typing import AnyDataFrame, ColumnName, UseStandardTagsDict
from woodwork.utils import _get_column_logical_type, _parse_logical_type, import_or_none

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


class WoodworkTableAccessor:
    def __init__(self, dataframe):
        self._dataframe_weakref = weakref.ref(dataframe)
        self._schema = None

    def init(self, **kwargs):
        """Initializes Woodwork typing information for a DataFrame with a partial schema.

        Logical type priority:
            1. Types specified in ``logical_types``
            2. Types specified in ``partial_schema``
            3. Types inferred by ``ww.type_system.infer_logical_type``

        Other Info priority:
            1. Parameter passed in
            2. Value specified in ``partial_schema``

        Args:
            schema (Woodwork.TableSchema, optional): Typing information to use for the DataFrame instead of performing inference.
                 Specified arguments will override the schema's typing information.
            index (str, optional): Name of the index column.
            time_index (str, optional): Name of the time index column.
            logical_types (Dict[str -> LogicalType], optional): Dictionary mapping column names in
                the DataFrame to the LogicalType for the column. Setting a column's logical type to None in this dict will
                force a logical to be inferred.
            ignore_columns (list[str] or set[str], optional): List of columns to ignore for inferring logical types. If a column name
                is included in this list, then it cannot be part of the logical_types dictionary argument, and it must be part
                of an existing schema for the dataframe. This argument can be used when a column has a logical type that has
                already been inferred and its physical dtype is not expected to have changed since its last inference.
            already_sorted (bool, optional): Indicates whether the input DataFrame is already sorted on the time
                index. If False, will sort the dataframe first on the time_index and then on the index (pandas DataFrame
                only). Defaults to False.
            name (str, optional): Name used to identify the DataFrame.
            semantic_tags (dict, optional): Dictionary mapping column names in Woodwork to the
                semantic tags for the column. The keys in the dictionary should be strings
                that correspond to column names. There are two options for
                specifying the dictionary values:
                (str): If only one semantic tag is being set, a single string can be used as a value.
                (list[str] or set[str]): If multiple tags are being set, a list or set of strings can be
                used as the value.
                Semantic tags will be set to an empty set for any column not included in the
                dictionary.
            table_metadata (Dict[str -> json serializable], optional): Dictionary containing extra metadata for Woodwork.
            column_metadata (Dict[str -> Dict[str -> json serializable]], optional): Dictionary mapping column names
                to that column's metadata dictionary.
            use_standard_tags (bool, Dict[str -> bool], optional): Determines whether standard semantic tags will be
                added to columns based on the specified logical type for the column.
                If a single boolean is supplied, will apply the same use_standard_tags value to all columns.
                A dictionary can be used to specify ``use_standard_tags`` values for individual columns.
                Unspecified columns will use the default value of True.
            column_descriptions (Dict[str -> str], optional): Dictionary mapping column names to column descriptions.
            column_origins (str, Dict[str -> str], optional): Origin of each column. If a string is supplied, it is
                used as the origin for all columns. A dictionary can be used to set origins for individual columns.
            validate (bool, optional): Whether parameter and data validation should occur. Defaults to True. Warning:
                Should be set to False only when parameters and data are known to be valid.
                Any errors resulting from skipping validation with invalid inputs may not be easily understood.
        """
        self.init_with_partial_schema(**kwargs)

    def init_with_full_schema(
        self,
        schema: TableSchema,
        validate: bool = True,
        **kwargs,
    ) -> None:
        """Initializes Woodwork typing information for a DataFrame with a complete schema.

        Args:
            schema (Woodwork.TableSchema): Typing information to use for the DataFrame instead of performing inference. Note that any changes made to the
                schema object after initialization will propagate to the DataFrame. Similarly, to avoid unintended typing information changes, the same schema
                object should not be shared between DataFrames.
            validate (bool, optional): Whether parameter and data validation should occur. Defaults to True. Warning: Should be set to False only when
                parameters and data are known to be valid. Any errors resulting from skipping validation with invalid inputs may not be easily understood.
        """
        if validate:
            _check_schema(self._dataframe, schema)
            _check_unique_column_names(self._dataframe)
        self._schema = schema

        extra_params = [key for key in kwargs]
        if extra_params:
            warnings.warn(
                "A schema was provided and the following parameters were ignored: "
                + ", ".join(extra_params),
                ParametersIgnoredWarning,
            )

    def init_with_partial_schema(
        self,
        schema: Optional[TableSchema] = None,
        index: Optional[str] = None,
        time_index: Optional[str] = None,
        logical_types: Optional[Dict[ColumnName, Union[str, LogicalType, None]]] = None,
        ignore_columns: Optional[List[str]] = None,
        already_sorted: Optional[bool] = False,
        name: Optional[str] = None,
        semantic_tags: Optional[
            Dict[ColumnName, Union[str, List[str], Set[str]]]
        ] = None,
        table_metadata: Optional[dict] = None,
        column_metadata: Optional[Dict[ColumnName, dict]] = None,
        use_standard_tags: Optional[Union[bool, UseStandardTagsDict]] = None,
        column_descriptions: Optional[Dict[ColumnName, str]] = None,
        column_origins: Optional[Union[str, Dict[ColumnName, str]]] = None,
        null_invalid_values: Optional[bool] = False,
        validate: Optional[bool] = True,
        **kwargs,
    ) -> None:
        """Initializes Woodwork typing information for a DataFrame with a partial schema.

        Logical type priority:
            1. Types specified in ``logical_types``
            2. Types specified in ``partial_schema``
            3. Types inferred by ``ww.type_system.infer_logical_type``

        Other Info priority:
            1. Parameter passed in
            2. Value specified in ``partial_schema``

        Args:
            schema (Woodwork.TableSchema, optional): Typing information to use for the DataFrame instead of performing inference.
                 Specified arguments will override the schema's typing information.
            index (str, optional): Name of the index column.
            time_index (str, optional): Name of the time index column.
            logical_types (Dict[str -> LogicalType], optional): Dictionary mapping column names in
                the DataFrame to the LogicalType for the column. Setting a column's logical type to None in this dict will
                force a logical to be inferred.
            ignore_columns (list[str] or set[str], optional): List of columns to ignore for inferring logical types. If a column name
                is included in this list, then it cannot be part of the logical_types dictionary argument, and it must be part
                of an existing schema for the dataframe. This argument can be used when a column has a logical type that has
                already been inferred and its physical dtype is not expected to have changed since its last inference.
            already_sorted (bool, optional): Indicates whether the input DataFrame is already sorted on the time
                index. If False, will sort the dataframe first on the time_index and then on the index (pandas DataFrame
                only). Defaults to False.
            name (str, optional): Name used to identify the DataFrame.
            semantic_tags (dict, optional): Dictionary mapping column names in Woodwork to the
                semantic tags for the column. The keys in the dictionary should be strings
                that correspond to column names. There are two options for
                specifying the dictionary values:
                (str): If only one semantic tag is being set, a single string can be used as a value.
                (list[str] or set[str]): If multiple tags are being set, a list or set of strings can be
                used as the value.
                Semantic tags will be set to an empty set for any column not included in the
                dictionary.
            table_metadata (Dict[str -> json serializable], optional): Dictionary containing extra metadata for Woodwork.
            column_metadata (Dict[str -> Dict[str -> json serializable]], optional): Dictionary mapping column names
                to that column's metadata dictionary.
            use_standard_tags (bool, Dict[str -> bool], optional): Determines whether standard semantic tags will be
                added to columns based on the specified logical type for the column.
                If a single boolean is supplied, will apply the same use_standard_tags value to all columns.
                A dictionary can be used to specify ``use_standard_tags`` values for individual columns.
                Unspecified columns will use the default value of True.
            column_descriptions (Dict[str -> str], optional): Dictionary mapping column names to column descriptions.
            column_origins (str, Dict[str -> str], optional): Origin of each column. If a string is supplied, it is
                used as the origin for all columns. A dictionary can be used to set origins for individual columns.
            validate (bool, optional): Whether parameter and data validation should occur. Defaults to True. Warning:
                Should be set to False only when parameters and data are known to be valid.
                Any errors resulting from skipping validation with invalid inputs may not be easily understood.
        """
        if validate:
            _validate_accessor_params(
                self._dataframe,
                index,
                time_index,
                logical_types,
                ignore_columns,
                schema,
                use_standard_tags,
            )

        existing_logical_types = {}
        existing_col_descriptions = {}
        existing_col_metadata = {}
        existing_use_standard_tags = {}
        existing_semantic_tags = {}
        existing_col_origins = {}

        if schema:  # pull schema parameters
            name = name if name is not None else schema.name
            index = index if index is not None else schema.index
            time_index = time_index if time_index is not None else schema.time_index
            table_metadata = table_metadata or schema.metadata
            for col_name, col_schema in schema.columns.items():
                existing_logical_types[col_name] = col_schema.logical_type
                existing_semantic_tags[col_name] = (
                    col_schema.semantic_tags
                    - {"time_index"}
                    - {"index"}
                    - col_schema.logical_type.standard_tags
                )
                existing_col_descriptions[col_name] = col_schema.description
                existing_col_origins[col_name] = col_schema.origin
                existing_col_metadata[col_name] = col_schema.metadata
                existing_use_standard_tags[col_name] = col_schema.use_standard_tags

        # overwrite schema parameters with specified kwargs
        logical_types = _infer_missing_logical_types(
            self._dataframe,
            logical_types,
            existing_logical_types,
            ignore_columns,
            null_invalid_values=null_invalid_values,
        )
        column_descriptions = {
            **existing_col_descriptions,
            **(column_descriptions or {}),
        }
        column_metadata = {**existing_col_metadata, **(column_metadata or {})}
        column_names = list(self._dataframe.columns)
        use_standard_tags = _merge_use_standard_tags(
            existing_use_standard_tags,
            use_standard_tags,
            column_names,
        )
        semantic_tags = {**existing_semantic_tags, **(semantic_tags or {})}
        column_origins = {**existing_col_origins, **(column_origins or {})}

        self._schema = TableSchema(
            column_names=column_names,
            logical_types=logical_types,
            name=name,
            index=index,
            time_index=time_index,
            semantic_tags=copy.deepcopy(semantic_tags),
            table_metadata=copy.deepcopy(table_metadata),
            column_metadata=copy.deepcopy(column_metadata),
            use_standard_tags=copy.deepcopy(use_standard_tags),
            column_descriptions=column_descriptions,
            column_origins=column_origins,
            validate=validate,
            **kwargs,
        )
        self._set_underlying_index()
        if self._schema.time_index is not None:
            self._sort_columns(already_sorted)

    def __eq__(self, other, deep=True):
        if not self._schema.__eq__(other.ww._schema, deep=deep):
            return False

        # Only check pandas DataFrames for equality
        if (
            deep
            and isinstance(self._dataframe, pd.DataFrame)
            and isinstance(other.ww._dataframe, pd.DataFrame)
        ):
            return self._dataframe.equals(other.ww._dataframe)
        return True

    @_check_table_schema
    def __getattr__(self, attr):
        # Called if method is not present on the Accessor
        # If the method is present on TableSchema, uses that method.
        # If the method is present on DataFrame, uses that method.
        if hasattr(self._schema, attr):
            return self._make_schema_call(attr)
        if hasattr(self._dataframe, attr):
            return self._make_dataframe_call(attr)
        else:
            raise AttributeError(f"Woodwork has no attribute '{attr}'")

    @_check_table_schema
    def __getitem__(self, key):
        if isinstance(key, list):
            columns = set(self._dataframe.columns)
            diff = list(set(key).difference(columns))

            if diff:
                raise ColumnNotPresentError(sorted(diff))

            return self._get_subset_df_with_schema(key)

        if key not in self._dataframe:
            raise ColumnNotPresentError(key)

        series = self._dataframe[key]
        column = copy.deepcopy(self._schema.columns[key])

        series.ww.init(schema=column, validate=False)

        return series

    def __setitem__(self, col_name, column):
        series = tuple(pkg.Series for pkg in (pd, dd, ps) if pkg)
        if not isinstance(column, series):
            raise ValueError("New column must be of Series type")

        if column.ww.schema is not None and "index" in column.ww.semantic_tags:
            warnings.warn(
                f'Cannot add "index" tag on {col_name} directly to the DataFrame. The "index" tag has been removed from {col_name}. To set this column as a Woodwork index, please use df.ww.set_index',
                IndexTagRemovedWarning,
            )
            column.ww.set_semantic_tags(column.ww.semantic_tags - {"index"})

        # Don't allow reassigning of index or time index with setitem
        if self.index == col_name:
            raise KeyError(
                "Cannot reassign index. Change column name and then use df.ww.set_index to reassign index.",
            )
        if self.time_index == col_name:
            raise KeyError(
                "Cannot reassign time index. Change column name and then use df.ww.set_time_index to reassign time index.",
            )

        if column.ww._schema is None:
            column = init_series(column, use_standard_tags=True)

        self._dataframe[col_name] = column
        self._schema.columns[col_name] = column.ww._schema

    def __repr__(self):
        """A string representation of a Woodwork table containing typing information"""
        return repr(self._get_typing_info())

    def _repr_html_(self):
        """An HTML representation of a Woodwork table for IPython.display in Jupyter Notebooks
        containing typing information and a preview of the data."""
        return self._get_typing_info().to_html()

    @_check_table_schema
    def _get_typing_info(self):
        """Creates a DataFrame that contains the typing information for a Woodwork table."""
        typing_info = self._schema._get_typing_info().copy()
        typing_info.insert(0, "Physical Type", pd.Series(self.physical_types))
        # Maintain the same column order used in the DataFrame
        typing_info = typing_info.loc[list(self._dataframe.columns), :]
        return typing_info

    @property
    @_check_table_schema
    def name(self):
        """Name of the DataFrame"""
        return self._schema.name

    @name.setter
    @_check_table_schema
    def name(self, name):
        """Set name of the DataFrame"""
        self._schema.name = name

    @property
    @_check_table_schema
    def metadata(self):
        """Metadata of the DataFrame"""
        return self._schema.metadata

    @metadata.setter
    @_check_table_schema
    def metadata(self, metadata):
        """Set metadata of the DataFrame"""
        self._schema.metadata = metadata

    @property
    def _dataframe(self):
        return self._dataframe_weakref()

    @property
    @_check_table_schema
    def iloc(self):
        """
        Integer-location based indexing for selection by position.
        ``.iloc[]`` is primarily integer position based (from ``0`` to
        ``length-1`` of the axis), but may also be used with a boolean array.

        If the selection result is a DataFrame or Series, Woodwork typing
        information will be initialized for the returned object when possible.

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
        return _iLocIndexer(self._dataframe)

    @property
    @_check_table_schema
    def loc(self):
        """
        Access a group of rows by label(s) or a boolean array.

        ``.loc[]`` is primarily label based, but may also be used with a
        boolean array.

        If the selection result is a DataFrame or Series, Woodwork typing
        information will be initialized for the returned object when possible.

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
        return _locIndexer(self._dataframe)

    @property
    def schema(self):
        """A copy of the Woodwork typing information for the DataFrame."""
        if self._schema:
            return copy.deepcopy(self._schema)

    @property
    @_check_table_schema
    def physical_types(self):
        """A dictionary containing physical types for each column"""
        return {
            col_name: self._schema.logical_types[col_name]._get_valid_dtype(
                type(self._dataframe[col_name]),
            )
            for col_name in self._dataframe.columns
        }

    @property
    @_check_table_schema
    def types(self):
        """DataFrame containing the physical dtypes, logical types and semantic
        tags for the schema."""
        return self._get_typing_info()

    @property
    @_check_table_schema
    def logical_types(self):
        """A dictionary containing logical types for each column"""
        return self._schema.logical_types

    @property
    @_check_table_schema
    def semantic_tags(self):
        """A dictionary containing semantic tags for each column"""
        return self._schema.semantic_tags

    @property
    @_check_table_schema
    def index(self):
        """The index column for the table"""
        return self._schema.index

    @property
    @_check_table_schema
    def time_index(self):
        """The time index column for the table"""
        return self._schema.time_index

    @property
    @_check_table_schema
    def use_standard_tags(self):
        """A dictionary containing the use_standard_tags setting for each column in the table"""
        return self._schema.use_standard_tags

    @_check_table_schema
    def set_index(self, new_index):
        """Sets the index column of the DataFrame. Adds the 'index' semantic tag to the column
        and clears the tag from any previously set index column.

        Setting a column as the index column will also cause any previously set standard
        tags for the column to be removed.

        Clears the DataFrame's index by passing in None.

        Args:
            new_index (str): The name of the column to set as the index
        """
        self._schema.set_index(new_index)

        if self._schema.index is not None:
            _check_index(self._dataframe, self._schema.index)
        self._set_underlying_index()

    @_check_table_schema
    def set_time_index(self, new_time_index):
        """Set the time index. Adds the 'time_index' semantic tag to the column and
        clears the tag from any previously set index column

        Args:
            new_time_index (str): The name of the column to set as the time index.
                If None, will remove the time_index.
        """
        self._schema.set_time_index(new_time_index)

    @_check_table_schema
    def set_types(
        self,
        logical_types=None,
        semantic_tags=None,
        retain_index_tags=True,
        null_invalid_values=False,
    ):
        """Update the logical type and semantic tags for any columns names in the provided types dictionaries,
        updating the Woodwork typing information for the DataFrame.

        Args:
            logical_types (Dict[str -> str], optional): A dictionary defining the new logical types for the
                specified columns.
            semantic_tags (Dict[str -> str/list/set], optional): A dictionary defining the new semantic_tags for the
                specified columns.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will replace all semantic tags any time a column's
                semantic tags or logical type changes. Defaults to True.
            null_invalid_values (bool, optional): If True, replaces any invalid values with null. Defaults to False.
        """
        logical_types = logical_types or {}
        logical_types = {
            col_name: _parse_logical_type(ltype, col_name)
            for col_name, ltype in logical_types.items()
        }

        self._schema.set_types(
            logical_types=logical_types,
            semantic_tags=semantic_tags,
            retain_index_tags=retain_index_tags,
        )
        # go through changed ltypes and update dtype if necessary
        for col_name, logical_type in logical_types.items():
            series = self._dataframe[col_name]
            updated_series = logical_type.transform(
                series,
                null_invalid_values=null_invalid_values,
            )
            if updated_series is not series:
                self._dataframe[col_name] = updated_series

    @_check_table_schema
    def select(self, include=None, exclude=None, return_schema=False):
        """Create a DataFrame with Woodwork typing information initialized
        that includes only columns whose Logical Type and semantic tags match
        conditions specified in the list of types and tags to include or exclude.
        Values for both ``include`` and ``exclude`` cannot be provided in a
        single call.

        If no matching columns are found, an empty DataFrame will be returned.

        Args:
            include (str or LogicalType or list[str or LogicalType]): Logical
                types, semantic tags to include in the DataFrame.
            exclude (str or LogicalType or list[str or LogicalType]): Logical
                types, semantic tags to exclude from the DataFrame.
            return_schema (bool): If True, return only the schema for the
                matching columns. Defaults to False

        Returns:
            DataFrame: The subset of the original DataFrame that matches the
            conditions specified by ``include`` or ``exclude``. Has Woodwork
            typing information initialized.
        """
        if include is not None and exclude is not None:
            raise ValueError(
                "Cannot specify values for both 'include' and 'exclude' in a single call.",
            )
        if include is None and exclude is None:
            raise ValueError("Must specify values for either 'include' or 'exclude'.")

        cols_to_include = self._schema._filter_cols(include, exclude)

        if return_schema:
            return self._schema.get_subset_schema(cols_to_include)
        return self._get_subset_df_with_schema(cols_to_include)

    @_check_table_schema
    def add_semantic_tags(self, semantic_tags):
        """Adds specified semantic tags to columns, updating the Woodwork typing information.
        Will retain any previously set values.

        Args:
            semantic_tags (Dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataFrame to the tags that should be added to the column's semantic tags
        """
        self._schema.add_semantic_tags(semantic_tags)

    @_check_table_schema
    def remove_semantic_tags(self, semantic_tags):
        """Remove the semantic tags for any column names in the provided semantic_tags
        dictionary, updating the Woodwork typing information. Including `index` or `time_index`
        tags will set the Woodwork index or time index to None for the DataFrame.

        Args:
            semantic_tags (Dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataFrame to the tags that should be removed from the column's semantic tags
        """
        self._schema.remove_semantic_tags(semantic_tags)

    @_check_table_schema
    def reset_semantic_tags(self, columns=None, retain_index_tags=False):
        """Reset the semantic tags for the specified columns to the default values.
        The default values will be either an empty set or a set of the standard tags
        based on the column logical type, controlled by the use_standard_tags property on each column.
        Column names can be provided as a single string, a list of strings or a set of strings.
        If columns is not specified, tags will be reset for all columns.

        Args:
            columns (str/list/set, optional): The columns for which the semantic tags should be reset.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will clear all semantic tags. Defaults to
                False.
        """
        self._schema.reset_semantic_tags(
            columns=columns,
            retain_index_tags=retain_index_tags,
        )

    @_check_table_schema
    def to_dictionary(self):
        """Get a dictionary representation of the Woodwork typing information.

        Returns:
            dict: Description of the typing information.
        """
        return typing_info_to_dict(self._dataframe)

    @_check_table_schema
    def to_disk(
        self,
        path,
        format="csv",
        filename=None,
        data_subdirectory="data",
        typing_info_filename="woodwork_typing_info.json",
        compression=None,
        profile_name=None,
        **kwargs,
    ):
        """Write Woodwork table to disk in the format specified by `format`, location specified by `path`.
        Path could be a local path or an S3 path.
        If writing to S3 a tar archive of files will be written.

        Note:
            When serializing to parquet, Woodwork typing information will be stored in the parquet file metadata,
            and not in a separate typing info JSON file. Any value provided for `typing_info_filename` will be ignored.

        Args:
            path (str): Location on disk to write to (will be created as a directory if it does not exist)
            format (str, optional): Format to use for writing Woodwork data. Defaults to csv. Possible values are: {'csv', 'pickle', 'parquet'}.
            filename (str, optional): Name to use for the saved data file. Will default to the name of the dataframe or "data" if not specified.
            data_subdirectory (str, optional): Optional subdirectory to append to `path`. Will default to "data" if not specified.
            typing_info_filename (str, optional): Optional filename to use for storing Woodwork typing information JSON data.
                Will default to "woodwork_typing_info.json" if not specified. Will be ignored if serializing to parquet.
            compression (str, optional): Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}. Defaults to None.
            profile_name (str, optional): Name of AWS profile to use, False to use an anonymous profile, or None. Defaults to None.
            kwargs (keywords, optional): Additional keyword arguments to pass as keywords arguments to the underlying serialization method or to specify AWS profile.
        """
        serializer_cls = get_serializer(format)
        serializer = serializer_cls(
            path=path,
            filename=filename,
            data_subdirectory=data_subdirectory,
            typing_info_filename=typing_info_filename,
        )
        serializer.serialize(
            self._dataframe,
            compression=compression,
            profile_name=profile_name,
            **kwargs,
        )

    def _sort_columns(self, already_sorted):
        if _is_dask_dataframe(self._dataframe) or _is_spark_dataframe(self._dataframe):
            already_sorted = True  # Skip sorting for Dask and Spark input
        if not already_sorted:
            sort_cols = [self._schema.time_index, self._schema.index]
            if self._schema.index is None:
                sort_cols = [self._schema.time_index]
            self._dataframe.sort_values(sort_cols, inplace=True)

    def _set_underlying_index(self):
        """Sets the index of the underlying DataFrame to match the index column as
        specified by the TableSchema. Does not change the underlying index if no Woodwork index is
        specified. Only sets underlying index for pandas DataFrames.
        """
        if isinstance(self._dataframe, pd.DataFrame) and self._schema.index is not None:
            self._dataframe.set_index(self._schema.index, drop=False, inplace=True)
            # Drop index name to not overlap with the original column
            self._dataframe.index.name = None

    def _make_schema_call(self, attr):
        """Forwards the requested attribute onto the schema object.
        Results are that of the Woodwork.TableSchema class."""
        schema_attr = getattr(self._schema, attr)

        if callable(schema_attr):

            def wrapper(*args, **kwargs):
                return schema_attr(*args, **kwargs)

            return wrapper
        return schema_attr

    def _make_dataframe_call(self, attr):
        """Forwards the requested attribute onto the dataframe object. Intercepts return value,
        attempting to initialize Woodwork with the current schema when a new DataFrame is returned.
        Confirms schema is still valid for the original DataFrame."""
        dataframe_attr = getattr(self._dataframe, attr)

        if callable(dataframe_attr):

            def wrapper(*args, **kwargs):
                # Make DataFrame call and intercept the result
                result = dataframe_attr(*args, **kwargs)

                # Try to initialize Woodwork with the existing schema
                if _is_dataframe(result):
                    invalid_schema_message = get_invalid_schema_message(
                        result,
                        self._schema,
                    )
                    if invalid_schema_message:
                        warnings.warn(
                            TypingInfoMismatchWarning().get_warning_message(
                                attr,
                                invalid_schema_message,
                                "DataFrame",
                            ),
                            TypingInfoMismatchWarning,
                        )
                    else:
                        copied_schema = self.schema
                        result.ww.init_with_full_schema(
                            schema=copied_schema,
                            validate=False,
                        )
                else:
                    # Confirm that the schema is still valid on original DataFrame
                    # Important for inplace operations
                    invalid_schema_message = get_invalid_schema_message(
                        self._dataframe,
                        self._schema,
                    )

                    if invalid_schema_message:
                        warnings.warn(
                            TypingInfoMismatchWarning().get_warning_message(
                                attr,
                                invalid_schema_message,
                                "DataFrame",
                            ),
                            TypingInfoMismatchWarning,
                        )
                        self._schema = None

                # Always return the results of the DataFrame operation whether or not Woodwork is initialized
                return result

            return wrapper
        # Directly return non-callable DataFrame attributes
        return dataframe_attr

    def _get_subset_df_with_schema(self, cols_to_include, inplace=False):
        """Creates a new DataFrame from a list of column names with Woodwork initialized,
        retaining all typing information and maintaining the DataFrame's column order.
        """
        if inplace:
            if _is_dask_dataframe(self._dataframe):
                raise ValueError("Drop inplace not supported for Dask")
            if _is_spark_dataframe(self._dataframe):
                raise ValueError("Drop inplace not supported for Spark")

        assert all([col_name in self._schema.columns for col_name in cols_to_include])

        new_schema = self._schema.get_subset_schema(cols_to_include)
        if inplace:
            cols_to_drop = [
                col_name
                for col_name in self._dataframe.columns
                if col_name not in cols_to_include
            ]
            self._dataframe.drop(cols_to_drop, axis="columns", inplace=True)
            self.init_with_full_schema(schema=new_schema, validate=False)
            return
        new_df = self._dataframe[cols_to_include]
        new_df.ww.init_with_full_schema(schema=new_schema, validate=False)

        return new_df

    @_check_table_schema
    def pop(self, column_name):
        """Return a Series with Woodwork typing information and remove it from the DataFrame.

        Args:
            column (str): Name of the column to pop.

        Returns:
            Series: Popped series with Woodwork initialized
        """
        if column_name not in self._dataframe.columns:
            raise ColumnNotPresentError(column_name)

        series = self._dataframe.pop(column_name)

        # Initialize Woodwork typing info for series
        series.ww.init(schema=self.schema.columns[column_name], validate=False)

        # Update schema to not include popped column
        del self._schema.columns[column_name]

        return series

    @_check_table_schema
    def drop(self, columns, inplace=False):
        """Drop specified columns from a DataFrame.

        Args:
            columns (str or list[str]): Column name or names to drop. Must be present in the DataFrame.
            inplace (bool): If False, return a copy. Otherwise, do operation inplace and return None.

        Returns:
            DataFrame or None: DataFrame with the specified columns removed, maintaining Woodwork typing information or None if inplace=True. Only possible for pandas dataframes.

        Note:
            This method is used for removing columns only. To remove rows with ``drop``, go through the
            DataFrame directly and then reinitialize Woodwork with ``DataFrame.ww.init``
            instead of calling ``DataFrame.ww.drop``.
        """
        if not isinstance(columns, (list, set)):
            columns = [columns]

        not_present = [col for col in columns if col not in self._dataframe.columns]
        if not_present:
            raise ColumnNotPresentError(not_present)
        return self._get_subset_df_with_schema(
            [col for col in self._dataframe.columns if col not in columns],
            inplace=inplace,
        )

    @_check_table_schema
    def rename(self, columns, inplace=False):
        """Renames columns in a DataFrame, maintaining Woodwork typing information.

        Args:
            columns (Dict[str -> str]): A dictionary mapping current column names to new column names.
            inplace (bool): If False, return a copy. Otherwise, do operation inplace and return None.

        Returns:
            DataFrame or None: DataFrame with the specified columns renamed, maintaining Woodwork typing information or None if inplace=True. Only possible for pandas dataframes.
        """

        new_schema = self._schema.rename(columns)
        if inplace:
            if _is_dask_dataframe(self._dataframe):
                raise ValueError("Rename inplace not supported for Dask")
            if _is_spark_dataframe(self._dataframe):
                raise ValueError("Rename inplace not supported for Spark")
            self._dataframe.rename(columns=columns, inplace=True)
            self.init_with_full_schema(schema=new_schema)
            return
        new_df = self._dataframe.rename(columns=columns)

        new_df.ww.init_with_full_schema(schema=new_schema, validate=False)
        return new_df

    @_check_table_schema
    def mutual_information_dict(
        self,
        num_bins=10,
        nrows=None,
        include_index=False,
        include_time_index=False,
        callback=None,
        extra_stats=False,
        min_shared=25,
        random_seed=0,
        max_nunique=6000,
    ):
        """
        Calculates mutual information between all pairs of columns in the DataFrame that
        support mutual information. Call woodwork.utils.get_valid_mi_types to see which Logical Types support
        mutual information.

        Args:
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical. Defaults to 10.
            nrows (int): The number of rows to sample for when determining mutual info.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for mutual information calculations.
                If False, the index column will not have mutual information calculated for it.
                Defaults to False.
            include_time_index (bool): If True, the column specified as the time index will be
                included for mutual information calculations.
                If False, the time index column will not have mutual information calculated for it.
                Defaults to False.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:
                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            extra_stats (bool):  If True, additional column "shared_rows"
                recording the number of shared non-null rows for a column
                pair will be included with the dataframe. Defaults to False.
            min_shared (int): The number of shared non-null rows needed to
                calculate. Less rows than this will be considered too sparse
                to measure accurately and will return a NaN value. Must be
                non-negative. Defaults to 25.
            random_seed (int): Seed for the random number generator. Defaults to 0.
            max_nunique (int): The total maximum number of unique values for all large categorical columns (> 800 unique values).
                Categorical columns will be dropped until this number is met or until there is only one large categorical column.
                Defaults to 6000.

        Returns:
            list(dict): A list containing dictionaries that have keys `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect dependency).
        """
        return _get_dependence_dict(
            dataframe=self._dataframe,
            measures=["mutual_info"],
            num_bins=num_bins,
            nrows=nrows,
            include_index=include_index,
            include_time_index=include_time_index,
            callback=callback,
            extra_stats=extra_stats,
            min_shared=min_shared,
            random_seed=random_seed,
            max_nunique=max_nunique,
        )

    def mutual_information(
        self,
        num_bins=10,
        nrows=None,
        include_index=False,
        include_time_index=False,
        callback=None,
        extra_stats=False,
        min_shared=25,
        random_seed=0,
        max_nunique=6000,
    ):
        """Calculates mutual information between all pairs of columns in the DataFrame that
        support mutual information. Call woodwork.utils.get_valid_mi_types to see which Logical Types support
        mutual information.

        Args:
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical. Defaults to 10.
            nrows (int): The number of rows to sample for when determining mutual info.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for mutual information calculations.
                If False, the index column will not have mutual information calculated for it.
                Defaults to False.
            include_time_index (bool): If True, the column specified as the time index will be
                included for mutual information calculations.
                If False, the time index column will not have mutual information calculated for it.
                Defaults to False.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:
                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            extra_stats (bool):  If True, additional column "shared_rows"
                recording the number of shared non-null rows for a column
                pair will be included with the dataframe. Defaults to False.
            min_shared (int): The number of shared non-null rows needed to
                calculate. Less rows than this will be considered too sparse
                to measure accurately and will return a NaN value. Must be
                non-negative. Defaults to 25.
            random_seed (int): Seed for the random number generator. Defaults to 0.
            max_nunique (int): The total maximum number of unique values for all large categorical columns (> 800 unique values).
                Categorical columns will be dropped until this number is met or until there is only one large categorical column.
                Defaults to 6000.

        Returns:
            pd.DataFrame: A DataFrame containing mutual information with columns `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect dependency).
        """
        mutual_info = self.mutual_information_dict(
            num_bins=num_bins,
            nrows=nrows,
            include_index=include_index,
            include_time_index=include_time_index,
            callback=callback,
            extra_stats=extra_stats,
            min_shared=min_shared,
            random_seed=random_seed,
            max_nunique=max_nunique,
        )
        return pd.DataFrame(mutual_info)

    @_check_table_schema
    def pearson_correlation_dict(
        self,
        nrows=None,
        include_index=False,
        include_time_index=False,
        callback=None,
        extra_stats=False,
        min_shared=25,
        random_seed=0,
    ):
        """
        Calculates Pearson correlation coefficient between all pairs of columns in the DataFrame that
        support correlation. Works with numeric and datetime data. Call woodwork.utils.get_valid_pearson_types to
        see which Logical Types are supported.

        Args:
            nrows (int): The number of rows to sample for when determining correlation.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for correlation calculations.
                If False, the index column will not have the Pearson correlation calculated for it.
                Defaults to False.
            include_time_index (bool): If True, the column specified as the time index will be
                included for correlation calculations.
                If False, the time index column will not have the Pearson correlation calculated for it.
                Defaults to False.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:
                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            extra_stats (bool):  If True, additional column "shared_rows"
                recording the number of shared non-null rows for a column
                pair will be included with the dataframe. Defaults to False.
            min_shared (int): The number of shared non-null rows needed to
                calculate. Less rows than this will be considered too sparse
                to measure accurately and will return a NaN value. Must be
                non-negative. Defaults to 25.
            random_seed (int): Seed for the random number generator. Defaults to 0.

        Returns:
            list(dict): A list containing dictionaries that have keys `column_1`,
            `column_2`, and `pearson` that is sorted in decending order by correlation coefficient.
            Correlation coefficient values are between -1 and 1.
        """
        return _get_dependence_dict(
            dataframe=self._dataframe,
            measures=["pearson"],
            nrows=nrows,
            include_index=include_index,
            include_time_index=include_time_index,
            callback=callback,
            extra_stats=extra_stats,
            min_shared=min_shared,
            random_seed=random_seed,
        )

    def pearson_correlation(
        self,
        nrows=None,
        include_index=False,
        include_time_index=False,
        callback=None,
        extra_stats=False,
        min_shared=25,
        random_seed=0,
    ):
        """Calculates Pearson correlation coefficient between all pairs of columns in the DataFrame that
        support correlation. Works with numeric and datetime data. Call woodwork.utils.get_valid_pearson_types to
        see which Logical Types are supported.

        Args:
            nrows (int): The number of rows to sample for when determining correlation.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for correlation calculations.
                If False, the index column will not have the Pearson correlation calculated for it.
                Defaults to False.
            include_time_index (bool): If True, the column specified as the time index will be
                included for correlation calculations.
                If False, the time index column will not have the Pearson correlation calculated for it.
                Defaults to False.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:
                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            extra_stats (bool):  If True, additional column "shared_rows"
                recording the number of shared non-null rows for a column
                pair will be included with the dataframe. Defaults to False.
            min_shared (int): The number of shared non-null rows needed to
                calculate. Less rows than this will be considered too sparse
                to measure accurately and will return a NaN value. Must be
                non-negative. Defaults to 25.
            random_seed (int): Seed for the random number generator. Defaults to 0.

        Returns:
            pd.DataFrame: A DataFrame containing Pearson correlation coefficients with columns `column_1`,
            `column_2`, and `pearson` that is sorted in decending order by correlation value.
            Pearson values are between -1 and 1, with 0 meaning no correlation.
        """
        pearson_dict = self.pearson_correlation_dict(
            nrows=nrows,
            include_index=include_index,
            include_time_index=include_time_index,
            callback=callback,
            extra_stats=extra_stats,
            min_shared=min_shared,
            random_seed=random_seed,
        )
        return pd.DataFrame(pearson_dict)

    @_check_table_schema
    def spearman_correlation_dict(
        self,
        nrows=None,
        include_index=False,
        include_time_index=False,
        callback=None,
        extra_stats=False,
        min_shared=25,
        random_seed=0,
    ):
        """
        Calculates Spearman correlation coefficient between all pairs of columns in the DataFrame that
        support correlation. Works with numeric, ordinal, and datetime data. Call woodwork.utils.get_valid_spearman_types to
        see which Logical Types are supported.

        Args:
            nrows (int): The number of rows to sample for when determining correlation.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for correlation calculations.
                If False, the index column will not have the Spearman correlation calculated for it.
                Defaults to False.
            include_time_index (bool): If True, the column specified as the time index will be
                included for correlation calculations.
                If False, the time index column will not have the Spearman correlation calculated for it.
                Defaults to False.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:
                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            extra_stats (bool):  If True, additional column "shared_rows"
                recording the number of shared non-null rows for a column
                pair will be included with the dataframe. Defaults to False.
            min_shared (int): The number of shared non-null rows needed to
                calculate. Less rows than this will be considered too sparse
                to measure accurately and will return a NaN value. Must be
                non-negative. Defaults to 25.
            random_seed (int): Seed for the random number generator. Defaults to 0.

        Returns:
            list(dict): A list containing dictionaries that have keys `column_1`,
            `column_2`, and `spearman` that is sorted in decending order by correlation coefficient.
            Correlation coefficient values are between -1 and 1.
        """
        return _get_dependence_dict(
            dataframe=self._dataframe,
            measures=["spearman"],
            nrows=nrows,
            include_index=include_index,
            include_time_index=include_time_index,
            callback=callback,
            extra_stats=extra_stats,
            min_shared=min_shared,
            random_seed=random_seed,
        )

    def spearman_correlation(
        self,
        nrows=None,
        include_index=False,
        include_time_index=False,
        callback=None,
        extra_stats=False,
        min_shared=25,
        random_seed=0,
    ):
        """Calculates Spearman correlation coefficient between all pairs of columns in the DataFrame that
        support correlation. Works with numeric, ordinal, and datetime data. Call woodwork.utils.get_valid_spearman_types to
        see which Logical Types are supported.

        Args:
            nrows (int): The number of rows to sample for when determining correlation.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for correlation calculations.
                If False, the index column will not have the Spearman correlation calculated for it.
                Defaults to False.
            include_time_index (bool): If True, the column specified as the time index will be
                included for correlation calculations.
                If False, the time index column will not have the Spearman correlation calculated for it.
                Defaults to False.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:
                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            extra_stats (bool):  If True, additional column "shared_rows"
                recording the number of shared non-null rows for a column
                pair will be included with the dataframe. Defaults to False.
            min_shared (int): The number of shared non-null rows needed to
                calculate. Less rows than this will be considered too sparse
                to measure accurately and will return a NaN value. Must be
                non-negative. Defaults to 25.
            random_seed (int): Seed for the random number generator. Defaults to 0.

        Returns:
            pd.DataFrame: A DataFrame containing Spearman correlation coefficients with columns `column_1`,
            `column_2`, and `spearman` that is sorted in decending order by correlation value.
            Spearman values are between -1 and 1, with 0 meaning no correlation.
        """
        spearman_dict = self.spearman_correlation_dict(
            nrows=nrows,
            include_index=include_index,
            include_time_index=include_time_index,
            callback=callback,
            extra_stats=extra_stats,
            min_shared=min_shared,
            random_seed=random_seed,
        )
        return pd.DataFrame(spearman_dict)

    @_check_table_schema
    def dependence_dict(
        self,
        measures="all",
        num_bins=10,
        nrows=None,
        include_index=False,
        include_time_index=False,
        callback=None,
        extra_stats=False,
        min_shared=25,
        random_seed=0,
        max_nunique=6000,
        target_col=None,
    ):
        """Calculates dependence measures between all pairs of columns in the DataFrame that
        support measuring dependence. Supports boolean, categorical, datetime, and numeric data.
        Call woodwork.utils.get_valid_mi_types and woodwork.utils.get_valid_pearson_types
        for complete lists of supported Logical Types.

        Args:
            dataframe (pd.DataFrame): Data containing Woodwork typing information
                from which to calculate dependence.
            measures (list or str): Which dependence measures to calculate.
                A list of measures can be provided to calculate multiple
                measures at once.  Valid measure strings:

                - "pearson": calculates the Pearson correlation coefficient
                - "mutual_info": calculates the mutual information between columns
                - "spearman": calculates the Spearman corerlation coefficient
                - "max":  max(abs(pearson), abs(spearman), mutual) for each pair of columns
                - "all": includes columns for "pearson", "mutual_info", "spearman", and "max"
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical. Defaults to 10. Pearson
                calculation does not use binning.
            nrows (int): The number of rows to sample for when determining dependence.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for measuring dependence.
                If False, the index column will not be considered. Defaults to False.
            include_time_index (bool): If True, the column specified as the time index will be
                included for measuring dependence.
                If False, the time index column will not be considered. Defaults to False.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:

                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            extra_stats (bool):  If True, additional column "shared_rows"
                recording the number of shared non-null rows for a column
                pair will be included with the dataframe. Defaults to False. If
                the "max" measure is being used, a "measure_used" column will be
                added that records whether Pearson or mutual information was the
                maximum dependence for a particular row.
            min_shared (int): The number of shared non-null rows needed to
                calculate. Less rows than this will be considered too sparse
                to measure accurately and will return a NaN value. Must be
                non-negative. Defaults to 25.
            random_seed (int): Seed for the random number generator. Defaults to 0.
            max_nunique (int): The total maximum number of unique values for all large categorical columns (> 800 unique values).
                Categorical columns will be dropped until this number is met or until there is only one large categorical column.
                Defaults to 6000.
            target_col (str): The column name of the target. If provided, will only calculate the dependence dictionary between other columns and this target column.
                The target column will be `column_2` in the returned result. Defaults to None.

        Returns:
            list(dict): A list containing dictionaries that have keys `column_1`,
            `column_2`, and keys for the specified dependence measures. The list is
            sorted in decending order by the first specified measure.
            Dependence information values are between 0 (no dependence) and 1
            (perfect dependency). For Pearson and Spearman, values range from -1 to 1 but 0 is
            still no dependence.
        """
        return _get_dependence_dict(
            dataframe=self._dataframe,
            measures=measures,
            num_bins=num_bins,
            nrows=nrows,
            include_index=include_index,
            include_time_index=include_time_index,
            callback=callback,
            extra_stats=extra_stats,
            min_shared=min_shared,
            random_seed=random_seed,
            max_nunique=max_nunique,
            target_col=target_col,
        )

    def dependence(
        self,
        measures="all",
        num_bins=10,
        nrows=None,
        include_index=False,
        include_time_index=False,
        callback=None,
        extra_stats=False,
        min_shared=25,
        random_seed=0,
        max_nunique=6000,
        target_col=None,
    ):
        """Calculates dependence measures between all pairs of columns in the DataFrame that
        support measuring dependence. Supports boolean, categorical, datetime, and numeric data.
        Call woodwork.utils.get_valid_mi_types and woodwork.utils.get_valid_pearson_types
        for complete lists of supported Logical Types.

        Args:
            dataframe (pd.DataFrame): Data containing Woodwork typing information
                from which to calculate dependence.
            measures (list or str): Which dependence measures to calculate.
                A list of measures can be provided to calculate multiple
                measures at once.  Valid measure strings:

                - "pearson": calculates the Pearson correlation coefficient
                - "mutual_info": calculates the mutual information between columns
                - "spearman": calculates the Spearman correlation coefficient
                - "max":  max(abs(pearson), abs(spearman), mutual) for each pair of columns
                - "all": includes columns for "pearson", "mutual_info", "spearman", and "max"
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical. Defaults to 10. Pearson
                calculation does not use binning.
            nrows (int): The number of rows to sample for when determining dependence.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for measuring dependence.
                If False, the index column will not be considered. Defaults to False.
            include_time_index (bool): If True, the column specified as the time index will be
                included for measuring dependence.
                If False, the time index column will not be considered. Defaults to False.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:

                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            extra_stats (bool):  If True, additional column "shared_rows"
                recording the number of shared non-null rows for a column
                pair will be included with the dataframe. Defaults to False. If
                the "max" measure is being used, a "measure_used" column will be
                added that records whether Pearson or mutual information was the
                maximum dependence for a particular row.
            min_shared (int): The number of shared non-null rows needed to
                calculate. Less rows than this will be considered too sparse
                to measure accurately and will return a NaN value. Must be
                non-negative. Defaults to 25.
            random_seed (int): Seed for the random number generator. Defaults to 0.
            max_nunique (int): The maximum number of unique values for large categorical columns (> 800 unique values).
                Categorical columns will be dropped until this number is met or until there is only one large categorical column.
                Defaults to 6000.
            target_col (str): The column name of the target. If provided, will only calculate the dependence dictionary between other columns and this target column.
                The target column will be `column_2` in the returned result. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with the columns `column_1`,
            `column_2`, and keys for the specified dependence measures. The rows
            are sorted in decending order by the first specified measure.
            Dependence information values are between 0 (no dependence) and 1
            (perfect dependency). For Pearson and Spearman, values range from -1 to 1 but 0 is
            still no dependence.  Additional columns will be included if the
            `extra_stats` is True.
        """
        dep_dict = _get_dependence_dict(
            dataframe=self._dataframe,
            measures=measures,
            num_bins=num_bins,
            nrows=nrows,
            include_index=include_index,
            include_time_index=include_time_index,
            callback=callback,
            extra_stats=extra_stats,
            min_shared=min_shared,
            random_seed=random_seed,
            max_nunique=max_nunique,
            target_col=target_col,
        )
        return pd.DataFrame(dep_dict)

    def get_valid_mi_columns(self, include_index=False, include_time_index=False):
        """Retrieves a list of columns from the DataFrame with valid Logical Types that support mutual
        information. Call woodwork.utils.get_valid_mi_types to see which Logical Types support mutual information.

        Args:
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for mutual information calculations.
                If False, the index column will not have mutual information calculated for it.
                Defaults to False.
            include_time_index (bool): If True, the column specified as the time index will be
                included for mutual information calculations.
                If False, the time index column will not have mutual information calculated for it.
                Defaults to False.

        Returns:
            list: A list of column names that have valid Logical Types that support mutual information.
        """
        return _get_valid_mi_columns(self._dataframe, include_index, include_time_index)

    @_check_table_schema
    def describe_dict(
        self,
        include: Sequence[Union[str, LogicalType]] = None,
        callback: Callable[[int, int, int, str, float], Any] = None,
        results_callback: Callable[[pd.DataFrame, pd.Series], Any] = None,
        extra_stats: bool = False,
        bins: int = 10,
        top_x: int = 10,
        recent_x: int = 10,
    ) -> Dict[str, dict]:
        """Calculates statistics for data contained in the DataFrame.

        Args:
            include (list[str or LogicalType], optional): filter for what columns to include in the
                statistics returned. Can be a list of column names, semantic tags, logical types, or a list
                combining any of the three. It follows the most broad specification. Favors logical types
                then semantic tag then column name. If no matching columns are found, an empty DataFrame
                will be returned.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:

                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            results_callback (callable, optional): function to be called with intermediate results. Has the following parameters:

                - results_so_far (pd.DataFrame): the full dataframe calculated so far
                - most_recent_calculation (pd.Series): the calculations for the most recent column

            extra_stats (bool): If True, will calculate a histogram for numeric columns, top values
                for categorical columns and value counts for the most recent values in datetime columns. Will also
                calculate value counts within the range of values present for integer columns if the range of
                values present is less than or equal to than the number of bins used to compute the histogram.
                Output can be controlled by bins, top_x and recent_x parameters.
            bins (int): Number of bins to use when calculating histogram for numeric columns. Defaults to 10.
                Will be ignored unless extra_stats=True.
            top_x (int): Number of items to return when getting the most frequently occurring values for categorical
                columns. Defaults to 10. Will be ignored unless extra_stats=True.
            recent_x (int): Number of values to return when calculating value counts for the most recent dates in
                datetime columns. Defaults to 10. Will be ignored unless extra_stats=True.

        Returns:
            Dict[str -> dict]: A dictionary with a key for each column in the data or for each column
            matching the logical types, semantic tags or column names specified in ``include``, paired
            with a value containing a dictionary containing relevant statistics for that column.
        """
        return _get_describe_dict(
            self._dataframe,
            include=include,
            callback=callback,
            results_callback=results_callback,
            extra_stats=extra_stats,
            bins=bins,
            top_x=top_x,
            recent_x=recent_x,
        )

    def describe(
        self,
        include: Sequence[Union[str, LogicalType]] = None,
        callback: Callable[[int, int, int, str, float], Any] = None,
        results_callback: Callable[[pd.DataFrame, pd.Series], Any] = None,
    ) -> pd.DataFrame:
        """Calculates statistics for data contained in the DataFrame.

        Args:
            include (list[str or LogicalType], optional): filter for what columns to include in the
                statistics returned. Can be a list of column names, semantic tags, logical types, or a list
                combining any of the three. It follows the most broad specification. Favors logical types
                then semantic tag then column name. If no matching columns are found, an empty DataFrame
                will be returned.
            callback (callable, optional): Function to be called with incremental updates. Has the following parameters:

                - update (int): change in progress since last call
                - progress (int): the progress so far in the calculations
                - total (int): the total number of calculations to do
                - unit (str): unit of measurement for progress/total
                - time_elapsed (float): total time in seconds elapsed since start of call
            results_callback (callable, optional): function to be called with intermediate results. Has the following parameters:

                - results_so_far (pd.DataFrame): the full dataframe calculated so far
                - most_recent_calculation (pd.Series): the calculations for the most recent column

        Returns:
            pd.DataFrame: A Dataframe containing statistics for the data or the subset of the original
            DataFrame that contains the logical types, semantic tags, or column names specified
            in ``include``.
        """
        results = self.describe_dict(
            include=include,
            callback=callback,
            results_callback=results_callback,
        )
        index_order = [
            "physical_type",
            "logical_type",
            "semantic_tags",
            "count",
            "nunique",
            "nan_count",
            "mean",
            "mode",
            "std",
            "min",
            "first_quartile",
            "second_quartile",
            "third_quartile",
            "max",
            "num_true",
            "num_false",
        ]
        return pd.DataFrame(results).reindex(index_order)

    @_check_table_schema
    def value_counts(self, ascending=False, top_n=10, dropna=False):
        """Returns a list of dictionaries with counts for the most frequent values in each column (only
            for columns with `category` as a standard tag).


        Args:
            ascending (bool): Defines whether each list of values should be sorted most frequent
                to least frequent value (False), or least frequent to most frequent value (True).
                Defaults to False.

            top_n (int): the number of top values to retrieve. Defaults to 10.

            dropna (bool): determines whether to remove NaN values when finding frequency. Defaults
                to False.

        Returns:
            list(dict): a list of dictionaries for each categorical column with keys `count`
            and `value`.
        """
        return _get_value_counts(self._dataframe, ascending, top_n, dropna)

    @_check_table_schema
    def infer_temporal_frequencies(self, temporal_columns=None, debug=False):
        """Infers the observation frequency (daily, biweekly, yearly, etc) of each temporal column
            in the DataFrame. Temporal columns are ones with the logical type Datetime or Timedelta.
            Not supported for Dask and Spark DataFrames.

        Args:
            temporal_columns (list[str], optional): Columns for which frequencies should be inferred. Must be columns
                that are present in the DataFrame and are temporal in nature. Defaults to None. If not
                specified, all temporal columns will have their frequencies inferred.
            debug (boolean): A flag used to determine if more information should be returned for each temporal column if
                no uniform frequency was found.

        Returns:
            (dict): A dictionary where each key is a temporal column from the DataFrame, and the
                value is its observation frequency represented as a pandas offset alias string (D, M, Y, etc.)
                or None if no uniform frequency was present in the data.

        Note:
            The pandas util ``pd.infer_freq``, which is used in this method, has the following behaviors:
                - If even one row in a column does not follow the frequency seen in the remaining rows,
                    no frequency will be inferred. Example of otherwise daily data that skips one day:
                    ``['2011-01-03', '2011-01-04', '2011-01-05', '2011-01-07']``.
                - If any NaNs are present in the data, no frequency will be inferred.
                - Pandas will use the largest offset alias available to it, so ``W`` will be inferred for weekly data instead of ``7D``.
                    The list of available offset aliases, which include aliases such as ``B`` for business day or ``N`` for nanosecond,
                    can be found at https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
                - Offset aliases can be combined to create something like ``2d1H``, which could also be expressed as '49H'.
                    Pandas' frequency inference will return the lower common alias, ``49H``, in situations when it'd otherwise
                    need to combine aliases.
                - Offset strings can contain more information than just the offset alias. For example, a date range
                    ``pd.date_range(start="2020-01-01", freq="w", periods=10)`` will be inferred to have frequency ``W-SUN``.
                    That string is an offset alias with an anchoring suffix that indicates that the data is not only
                    observed at a weekly frequency, but that all the dates are on Sundays. More anchored offsets
                    can be seen here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets
                - Some frequencies that can be defined for a ``pd.date_range`` cannot then be re-inferred by pandas' ``pd.infer_freq``.
                    One example of this can be seen when using the business day offset alias ``B``
                    ``pd.date_range(start="2020-01-01", freq="4b", periods=10)``, which is a valid ``freq``
                    parameter when building the date range, but is not then inferrable.

            The algorithm used to infer frequency on noisy data can configured (see https://woodwork.alteryx.com/en/stable/guides/setting_config_options.html#Available-Config-Settings)

        """
        return _infer_temporal_frequencies(
            self._dataframe,
            temporal_columns=temporal_columns,
            debug=debug,
        )

    @_check_table_schema
    def validate_logical_types(self, return_invalid_values=False):
        """Validates the dataframe based on the logical types.
        If a column's dtype does not match its logical type's required dtype,
        will raise a TypeValidationError even when return_invalid_indices is True.

        Args:
            return_invalid_values (bool): Whether or not to return invalid data values

        Returns:
            DataFrame: If return_invalid_values is True, returns invalid data values.
        """
        invalid_values = []
        for column in self.columns:
            series = self.ww[column]
            values = series.ww.validate_logical_type(
                return_invalid_values=return_invalid_values,
            )
            if values is not None:
                invalid_values.append(values)

        if return_invalid_values and invalid_values:
            concat = pd.concat
            if _is_dask_dataframe(self._dataframe):
                concat = dd.concat
            if _is_spark_dataframe(self._dataframe):
                concat = ps.concat

            return concat(invalid_values, axis=1)


def _validate_accessor_params(
    dataframe,
    index,
    time_index,
    logical_types,
    ignore_columns,
    schema,
    use_standard_tags,
) -> None:
    _check_unique_column_names(dataframe)
    if use_standard_tags is not None:
        _check_use_standard_tags(use_standard_tags)
    if schema is not None:
        _check_partial_schema(dataframe, schema)
        if index is None:
            # if no index was passed in as a parameter we need to validate the existing index
            index = schema.index
    if index is not None:
        _check_index(dataframe, index)
    if logical_types:
        _check_logical_types(dataframe.columns, logical_types)
    if ignore_columns:
        _check_ignore_columns(dataframe.columns, logical_types, schema, ignore_columns)
    if time_index is not None:
        datetime_format = None
        logical_type = None
        if logical_types is not None and time_index in logical_types:
            logical_type = logical_types[time_index]
            if type(logical_types[time_index]) == Datetime:
                datetime_format = logical_types[time_index].datetime_format

        _check_time_index(
            dataframe,
            time_index,
            datetime_format=datetime_format,
            logical_type=logical_type,
        )


def _check_unique_column_names(dataframe):
    if not dataframe.columns.is_unique:
        raise IndexError("Dataframe cannot contain duplicate columns names")


def _check_index(dataframe, index):
    if index not in dataframe.columns:
        # User specifies an index that is not in the dataframe
        raise ColumnNotPresentError(
            f"Specified index column `{index}` not found in dataframe",
        )
    if index is not None and isinstance(dataframe, pd.DataFrame):
        # User specifies a dataframe index that is not unique or contains null values
        # Does not check Dask dataframes to avoid pulling data into memory and Dask does not support is_unique
        if not dataframe[index].is_unique:
            raise IndexError("Index column must be unique")

        if dataframe[index].isnull().any():
            raise IndexError("Index contains null values")


def _check_time_index(dataframe, time_index, datetime_format=None, logical_type=None):
    if time_index not in dataframe.columns:
        raise ColumnNotPresentError(
            f"Specified time index column `{time_index}` not found in dataframe",
        )
    if not (
        _is_numeric_series(dataframe[time_index], logical_type)
        or col_is_datetime(dataframe[time_index], datetime_format=datetime_format)
    ):
        raise TypeError("Time index column must contain datetime or numeric values")


def _check_logical_types(dataframe_columns, logical_types):
    if not isinstance(logical_types, dict):
        raise TypeError("logical_types must be a dictionary")
    cols_not_found = set(logical_types.keys()).difference(set(dataframe_columns))
    if cols_not_found:
        raise ColumnNotPresentError(
            "logical_types contains columns that are not present in "
            f"dataframe: {sorted(list(cols_not_found))}",
        )


def _check_ignore_columns(dataframe_columns, logical_types, schema, ignore_columns):
    if not isinstance(ignore_columns, (list, set)):
        raise TypeError("ignore_columns must be a list or set")
    cols_not_found = set(ignore_columns).difference(set(dataframe_columns))
    if cols_not_found:
        raise ColumnNotPresentError(
            "ignore_columns contains columns that are not present in "
            f"dataframe: {sorted(list(cols_not_found))}",
        )
    if logical_types:
        col_ignored_and_set = set(logical_types.keys()).intersection(
            set(ignore_columns),
        )
        if col_ignored_and_set:
            raise ColumnBothIgnoredAndSetError(
                "ignore_columns contains columns that are being set "
                f"in logical_types: {list(col_ignored_and_set)}",
            )
    if schema is None:
        raise WoodworkNotInitError(
            "ignore_columns cannot be set when the dataframe has no existing "
            "schema.",
        )


def _check_schema(dataframe, schema):
    if not isinstance(schema, TableSchema):
        raise TypeError("Provided schema must be a Woodwork.TableSchema object.")
    invalid_schema_message = get_invalid_schema_message(dataframe, schema)
    if invalid_schema_message:
        raise ValueError(
            f"Woodwork typing information is not valid for this DataFrame: {invalid_schema_message}",
        )


def _check_partial_schema(dataframe, schema: TableSchema) -> None:
    if not isinstance(schema, TableSchema):
        raise TypeError("Provided schema must be a Woodwork.TableSchema object.")
    dataframe_cols = set(dataframe.columns)
    schema_cols = set(schema.columns.keys())
    schema_cols_not_in_df = schema_cols - dataframe_cols
    if schema_cols_not_in_df:
        raise ColumnNotPresentError(
            f"The following columns in the typing information were missing from the DataFrame: "
            f"{schema_cols_not_in_df}",
        )


def _check_use_standard_tags(use_standard_tags):
    if not isinstance(use_standard_tags, (bool, dict)):
        raise TypeError("use_standard_tags must be a dictionary or a boolean")


def _infer_missing_logical_types(
    dataframe: AnyDataFrame,
    force_logical_types: Optional[Dict[ColumnName, Union[str, LogicalType]]] = None,
    existing_logical_types: Optional[Dict[ColumnName, Union[str, LogicalType]]] = None,
    ignore_columns: Optional[List[str]] = None,
    null_invalid_values: bool = False,
):
    """Performs type inference and updates underlying data"""
    force_logical_types = force_logical_types or {}
    existing_logical_types = existing_logical_types or {}
    ignore_columns = ignore_columns or []
    parsed_logical_types = {}
    for name in dataframe.columns:
        if name in ignore_columns and name in existing_logical_types:
            parsed_logical_types[name] = existing_logical_types.get(name)
            continue
        logical_type = (
            force_logical_types.get(name)
            if name in force_logical_types
            else existing_logical_types.get(name)
        )
        if name is not None:
            series = dataframe[name]
        elif name is None and len(dataframe.columns) == 1:
            series = dataframe.iloc[:, 0]
        parsed_logical_types[name] = _get_column_logical_type(
            series,
            logical_type,
            name,
        )
        updated_series = parsed_logical_types[name].transform(
            series,
            null_invalid_values=null_invalid_values,
        )
        if updated_series is not series:
            # NotImplementedError thrown by dask when attempting to re-initialize
            # data after being assigned a numeric column name
            try:
                dataframe[name] = updated_series
            except NotImplementedError:
                pass
    return parsed_logical_types


def _merge_use_standard_tags(
    existing_use_standard_tags: UseStandardTagsDict = None,
    use_standard_tags: Union[bool, UseStandardTagsDict] = None,
    column_names: Iterable[ColumnName] = None,
    default_use_standard_tag: bool = True,
) -> UseStandardTagsDict:
    """Combines existing and kwarg use_standard_tags and returns a UseStandardTagsDict with all column names
    Priority when merging:
    1. use_standard tags
    2. existing_use_standard_tags
    3. default_use_standard_tag
    """
    column_names = column_names or []
    if isinstance(use_standard_tags, bool):
        use_standard_tags = {col_name: use_standard_tags for col_name in column_names}
    else:
        use_standard_tags = {
            **{col_name: default_use_standard_tag for col_name in column_names},
            **(existing_use_standard_tags or {}),
            **(use_standard_tags or {}),
        }
    return use_standard_tags


@pd.api.extensions.register_dataframe_accessor("ww")
class PandasTableAccessor(WoodworkTableAccessor):
    pass


if dd:

    @dd.extensions.register_dataframe_accessor("ww")
    class DaskTableAccessor(WoodworkTableAccessor):
        pass


if ps:
    from pyspark.pandas.extensions import register_dataframe_accessor

    @register_dataframe_accessor("ww")
    class SparkTableAccessor(WoodworkTableAccessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not ps.get_option("compute.ops_on_diff_frames"):
                ps.set_option("compute.ops_on_diff_frames", True)
