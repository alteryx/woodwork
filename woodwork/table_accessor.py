import warnings

import pandas as pd

import woodwork.serialize as serialize
from woodwork.accessor_utils import (
    _get_valid_dtype,
    _is_dataframe,
    _update_column_dtype,
    init_series
)
from woodwork.exceptions import (
    ColumnNameMismatchWarning,
    ColumnNotPresentError,
    ParametersIgnoredWarning,
    StandardTagsChangedWarning,
    TypingInfoMismatchWarning
)
from woodwork.indexers import _iLocIndexer, _locIndexer
from woodwork.logical_types import Datetime
from woodwork.schema import Schema
from woodwork.statistics_utils import (
    _get_describe_dict,
    _get_mutual_information_dict,
    _get_value_counts
)
from woodwork.type_sys.utils import (
    _get_ltype_class,
    _is_numeric_series,
    col_is_datetime
)
from woodwork.utils import (
    _get_column_logical_type,
    _parse_logical_type,
    import_or_none,
    import_or_raise
)

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


class WoodworkTableAccessor:
    def __init__(self, dataframe):
        self._dataframe = dataframe
        self._schema = None

    def init(self,
             index=None,
             time_index=None,
             logical_types=None,
             make_index=False,
             already_sorted=False,
             schema=None,
             **kwargs):
        """Initializes Woodwork typing information for a DataFrame.

        Args:
            index (str, optional): Name of the index column.
            time_index (str, optional): Name of the time index column.
            logical_types (dict[str -> LogicalType]): Dictionary mapping column names in
                the DataFrame to the LogicalType for the column.
            make_index (bool, optional): If True, will create a new unique, numeric index column with the
                name specified by ``index`` and will add the new index column to the supplied DataFrame.
                If True, the name specified in ``index`` cannot match an existing column name in
                ``dataframe``. If False, the name is specified in ``index`` must match a column
                present in the ``dataframe``. Defaults to False.
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
            table_metadata (dict[str -> json serializable], optional): Dictionary containing extra metadata for Woodwork.
            column_metadata (dict[str -> dict[str -> json serializable]], optional): Dictionary mapping column names
                to that column's metadata dictionary.
            use_standard_tags (bool, optional): If True, will add standard semantic tags to columns based
                specified logical type for the column. Defaults to True.
            column_descriptions (dict[str -> str], optional): Dictionary mapping column names to column descriptions.
            schema (Woodwork.Schema, optional): Typing information to use for the DataFrame instead of performing inference.
                Any other arguments provided will be ignored. Note that any changes made to the schema object after
                initialization will propagate to the DataFrame. Similarly, to avoid unintended typing information changes,
                the same schema object should not be shared between DataFrames.
        """
        _validate_accessor_params(self._dataframe, index, make_index, time_index, logical_types, schema)
        if schema is not None:
            self._schema = schema
            extra_params = []
            if index is not None:
                extra_params.append('index')
            if make_index:
                extra_params.append('make_index')
            if time_index is not None:
                extra_params.append('time_index')
            if logical_types is not None:
                extra_params.append('logical_types')
            if already_sorted:
                extra_params.append('already_sorted')
            for key in kwargs:
                extra_params.append(key)
            if extra_params:
                warnings.warn("A schema was provided and the following parameters were ignored: " + ", ".join(extra_params), ParametersIgnoredWarning)

        else:
            if make_index:
                _make_index(self._dataframe, index)

            # Perform type inference and update underlying data
            parsed_logical_types = {}
            for name in self._dataframe.columns:
                series = self._dataframe[name]

                logical_type = None
                if logical_types:
                    logical_type = logical_types.get(name)

                logical_type = _get_column_logical_type(series, logical_type, name)
                parsed_logical_types[name] = logical_type

                updated_series = _update_column_dtype(series, logical_type)
                if updated_series is not series:
                    self._dataframe[name] = updated_series

            column_names = list(self._dataframe.columns)
            self._schema = Schema(column_names=column_names,
                                  logical_types=parsed_logical_types,
                                  index=index,
                                  time_index=time_index, **kwargs)

            self._set_underlying_index()
            if self._schema.time_index is not None:
                self._sort_columns(already_sorted)

    def __eq__(self, other):
        if self._schema != other.ww._schema:
            return False

        # Only check pandas DataFrames for equality
        if isinstance(self._dataframe, pd.DataFrame) and isinstance(other.ww._dataframe, pd.DataFrame):
            return self._dataframe.equals(other.ww._dataframe)
        return True

    def __getattr__(self, attr):
        # If the method is present on the Accessor, uses that method.
        # If the method is present on Schema, uses that method.
        # If the method is present on DataFrame, uses that method.
        if self._schema is None:
            _raise_init_error()
        if hasattr(self._schema, attr):
            return self._make_schema_call(attr)
        if hasattr(self._dataframe, attr):
            return self._make_dataframe_call(attr)
        else:
            raise AttributeError(f"Woodwork has no attribute '{attr}'")

    def __getitem__(self, key):
        if self._schema is None:
            _raise_init_error()
        if isinstance(key, list):
            columns = set(self._dataframe.columns)
            diff = list(set(key).difference(columns))

            if diff:
                raise ColumnNotPresentError(sorted(diff))

            return self._get_subset_df_with_schema(key, use_dataframe_order=False)

        if key not in self._dataframe:
            raise ColumnNotPresentError(key)

        series = self._dataframe[key]
        column = self.schema.columns[key]
        column['semantic_tags'] -= {'index', 'time_index'}
        series.ww.init(**column, use_standard_tags=self.use_standard_tags)
        return series

    def __setitem__(self, col_name, column):
        series = tuple(pkg.Series for pkg in (pd, dd, ks) if pkg)
        if not isinstance(column, series):
            raise ValueError('New column must be of Series type')

        # Don't allow reassigning of index or time index with setitem
        if self.index == col_name:
            raise KeyError('Cannot reassign index. Change column name and then use df.ww.set_index to reassign index.')
        if self.time_index == col_name:
            raise KeyError('Cannot reassign time index. Change column name and then use df.ww.set_time_index to reassign time index.')

        if column.name is not None and column.name != col_name:
            warnings.warn(ColumnNameMismatchWarning().get_warning_message(column.name, col_name),
                          ColumnNameMismatchWarning)

        if column.ww._schema is None:
            column = init_series(column, use_standard_tags=self.use_standard_tags)
        elif self.use_standard_tags and not column.ww.use_standard_tags:
            column.ww._schema['semantic_tags'] |= column.ww.logical_type.standard_tags
            message = StandardTagsChangedWarning().get_warning_message(self.use_standard_tags, col_name)
            warnings.warn(message, StandardTagsChangedWarning)
        elif not self.use_standard_tags and column.ww.use_standard_tags:
            column.ww._schema['semantic_tags'] -= column.ww.logical_type.standard_tags
            message = StandardTagsChangedWarning().get_warning_message(self.use_standard_tags, col_name)
            warnings.warn(message, StandardTagsChangedWarning)

        self._dataframe[col_name] = column
        self._schema.columns[col_name] = column.ww._schema

    def __repr__(self):
        """A string representation of a Woodwork table containing typing information"""
        return repr(self._get_typing_info())

    def _repr_html_(self):
        """An HTML representation of a Woodwork table for IPython.display in Jupyter Notebooks
        containing typing information and a preview of the data."""
        return self._get_typing_info().to_html()

    def _get_typing_info(self):
        """Creates a DataFrame that contains the typing information for a Woodwork table."""
        if self._schema is None:
            _raise_init_error()
        typing_info = self._schema._get_typing_info().copy()
        typing_info.insert(0, 'Physical Type', pd.Series(self.physical_types))
        # Maintain the same column order used in the DataFrame
        typing_info = typing_info.loc[list(self._dataframe.columns), :]
        return typing_info

    @property
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
        if self._schema is None:
            _raise_init_error()
        return _iLocIndexer(self._dataframe)

    @property
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
        if self._schema is None:
            _raise_init_error()
        return _locIndexer(self._dataframe)

    @property
    def schema(self):
        """A copy of the Woodwork typing information for the DataFrame."""
        if self._schema:
            return self._schema._get_subset_schema(list(self._dataframe.columns))

    @property
    def physical_types(self):
        """A dictionary containing physical types for each column"""
        if self._schema is None:
            _raise_init_error()
        return {col_name: _get_valid_dtype(type(self._dataframe[col_name]), self.schema.logical_types[col_name]) for col_name in self._dataframe.columns}

    @property
    def types(self):
        """DataFrame containing the physical dtypes, logical types and semantic
        tags for the Schema."""
        if self._schema is None:
            _raise_init_error()
        return self._get_typing_info()

    @property
    def logical_types(self):
        """A dictionary containing logical types for each column"""
        if self._schema is None:
            _raise_init_error()
        return self._schema.logical_types

    @property
    def semantic_tags(self):
        """A dictionary containing semantic tags for each column"""
        if self._schema is None:
            _raise_init_error()
        return self._schema.semantic_tags

    @property
    def index(self):
        """The index column for the table"""
        if self._schema is None:
            _raise_init_error()
        return self._schema.index

    @property
    def time_index(self):
        """The time index column for the table"""
        if self._schema is None:
            _raise_init_error()
        return self._schema.time_index

    def set_index(self, new_index):
        """Sets the index column of the DataFrame. Adds the 'index' semantic tag to the column
        and clears the tag from any previously set index column.

        Setting a column as the index column will also cause any previously set standard
        tags for the column to be removed.

        Clears the DataFrame's index by passing in None.

        Args:
            new_index (str): The name of the column to set as the index
        """
        if self._schema is None:
            _raise_init_error()
        self._schema.set_index(new_index)

        if self._schema.index is not None:
            _check_index(self._dataframe, self._schema.index)
        self._set_underlying_index()

    def set_time_index(self, new_time_index):
        """Set the time index. Adds the 'time_index' semantic tag to the column and
        clears the tag from any previously set index column

        Args:
            new_time_index (str): The name of the column to set as the time index.
                If None, will remove the time_index.
        """
        if self._schema is None:
            _raise_init_error()
        self._schema.set_time_index(new_time_index)

    def set_types(self, logical_types=None, semantic_tags=None, retain_index_tags=True):
        """Update the logical type and semantic tags for any columns names in the provided types dictionaries,
        updating the Woodwork typing information for the DataFrame.

        Args:
            logical_types (dict[str -> str], optional): A dictionary defining the new logical types for the
                specified columns.
            semantic_tags (dict[str -> str/list/set], optional): A dictionary defining the new semantic_tags for the
                specified columns.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will replace all semantic tags any time a column's
                semantic tags or logical type changes. Defaults to True.
        """
        if self._schema is None:
            _raise_init_error()
        logical_types = logical_types or {}
        logical_types = {col_name: _parse_logical_type(ltype, col_name) for col_name, ltype in logical_types.items()}

        self._schema.set_types(logical_types=logical_types,
                               semantic_tags=semantic_tags,
                               retain_index_tags=retain_index_tags)
        # go through changed ltypes and update dtype if necessary
        for col_name, logical_type in logical_types.items():
            series = self._dataframe[col_name]
            updated_series = _update_column_dtype(series, logical_type)
            if updated_series is not series:
                self._dataframe[col_name] = updated_series

    def select(self, include):
        """Create a DataFrame with Woodwork typing information initialized
        that includes only columns whose Logical Type and semantic tags are
        specified in the list of types and tags to include.

        If no matching columns are found, an empty DataFrame will be returned.

        Args:
            include (str or LogicalType or list[str or LogicalType]): Logical
                types, semantic tags to include in the DataFrame.

        Returns:
            DataFrame: The subset of the original DataFrame that contains just the
            logical types and semantic tags in ``include``. Has Woodwork typing
            information initialized.
        """
        if self._schema is None:
            _raise_init_error()
        cols_to_include = self._schema._filter_cols(include)
        return self._get_subset_df_with_schema(cols_to_include)

    def add_semantic_tags(self, semantic_tags):
        """Adds specified semantic tags to columns, updating the Woodwork typing information.
        Will retain any previously set values.

        Args:
            semantic_tags (dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataFrame to the tags that should be added to the column's semantic tags
        """
        if self._schema is None:
            _raise_init_error()
        self._schema.add_semantic_tags(semantic_tags)

    def remove_semantic_tags(self, semantic_tags):
        """Remove the semantic tags for any column names in the provided semantic_tags
        dictionary, updating the Woodwork typing information. Including `index` or `time_index`
        tags will set the Woodwork index or time index to None for the DataFrame.

        Args:
            semantic_tags (dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataFrame to the tags that should be removed from the column's semantic tags
        """
        if self._schema is None:
            _raise_init_error()
        self._schema.remove_semantic_tags(semantic_tags)

    def reset_semantic_tags(self, columns=None, retain_index_tags=False):
        """Reset the semantic tags for the specified columns to the default values.
        The default values will be either an empty set or a set of the standard tags
        based on the column logical type, controlled by the use_standard_tags property on the table.
        Column names can be provided as a single string, a list of strings or a set of strings.
        If columns is not specified, tags will be reset for all columns.

        Args:
            columns (str/list/set, optional): The columns for which the semantic tags should be reset.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will clear all semantic tags. Defaults to
                False.
        """
        if self._schema is None:
            _raise_init_error()
        self._schema.reset_semantic_tags(columns=columns, retain_index_tags=retain_index_tags)

    def to_dictionary(self):
        """Get a dictionary representation of the Woodwork typing information.

        Returns:
            dict: Description of the typing information.
        """
        if self._schema is None:
            _raise_init_error()
        return serialize.typing_info_to_dict(self._dataframe)

    def to_csv(self, path, sep=',', encoding='utf-8', engine='python', compression=None, profile_name=None):
        """Write Woodwork table to disk in the CSV format, location specified by `path`.
            Path could be a local path or a S3 path.
            If writing to S3 a tar archive of files will be written.

            Args:
                path (str) : Location on disk to write to (will be created as a directory)
                sep (str) : String of length 1. Field delimiter for the output file.
                encoding (str) : A string representing the encoding to use in the output file, defaults to 'utf-8'.
                engine (str) : Name of the engine to use. Possible values are: {'c', 'python'}.
                compression (str) : Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}.
                profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        """
        if self._schema is None:
            _raise_init_error()
        serialize.write_woodwork_table(self._dataframe, path, format='csv', index=False,
                                       sep=sep, encoding=encoding, engine=engine,
                                       compression=compression, profile_name=profile_name)

    def to_pickle(self, path, compression=None, profile_name=None):
        """Write Woodwork table to disk in the pickle format, location specified by `path`.
            Path could be a local path or a S3 path.
            If writing to S3 a tar archive of files will be written.

            Args:
                path (str) : Location on disk to write to (will be created as a directory)
                compression (str) : Name of the compression to use. Possible values are: {'gzip', 'bz2', 'zip', 'xz', None}.
                profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        """
        if self._schema is None:
            _raise_init_error()
        serialize.write_woodwork_table(self._dataframe, path, format='pickle',
                                       compression=compression, profile_name=profile_name)

    def to_parquet(self, path, compression=None, profile_name=None):
        """Write Woodwork table to disk in the parquet format, location specified by `path`.

            Path could be a local path or a S3 path.

            If writing to S3 a tar archive of files will be written.

            Note:
                As the engine `fastparquet` cannot handle nullable pandas dtypes, `pyarrow` will be used
                for serialization to parquet.

            Args:
                path (str): location on disk to write to (will be created as a directory)
                compression (str) : Name of the compression to use. Possible values are: {'snappy', 'gzip', 'brotli', None}.
                profile_name (str) : Name of AWS profile to use, False to use an anonymous profile, or None.
        """
        if self._schema is None:
            _raise_init_error()
        import_error_message = (
            "The pyarrow library is required to serialize to parquet.\n"
            "Install via pip:\n"
            "    pip install pyarrow\n"
            "Install via conda:\n"
            "   conda install pyarrow -c conda-forge"
        )
        import_or_raise('pyarrow', import_error_message)
        serialize.write_woodwork_table(self._dataframe, path, format='parquet',
                                       engine='pyarrow', compression=compression,
                                       profile_name=profile_name)

    def _sort_columns(self, already_sorted):
        if dd and isinstance(self._dataframe, dd.DataFrame) or (ks and isinstance(self._dataframe, ks.DataFrame)):
            already_sorted = True  # Skip sorting for Dask and Koalas input
        if not already_sorted:
            sort_cols = [self._schema.time_index, self._schema.index]
            if self._schema.index is None:
                sort_cols = [self._schema.time_index]
            self._dataframe.sort_values(sort_cols, inplace=True)

    def _set_underlying_index(self):
        """Sets the index of the underlying DataFrame to match the index column as
        specified by the Schema. Does not change the underlying index if no Woodwork index is
        specified. Only sets underlying index for pandas DataFrames.
        """
        if isinstance(self._dataframe, pd.DataFrame) and self._schema.index is not None:
            self._dataframe.set_index(self._schema.index, drop=False, inplace=True)
            # Drop index name to not overlap with the original column
            self._dataframe.index.name = None

    def _make_schema_call(self, attr):
        """Forwards the requested attribute onto the schema object.
        Results are that of the Woodwork.Schema class."""
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

                # Try to initialize Woodwork with the existing Schema
                if _is_dataframe(result):
                    invalid_schema_message = _get_invalid_schema_message(result, self._schema)
                    if invalid_schema_message:
                        warnings.warn(TypingInfoMismatchWarning().get_warning_message(attr, invalid_schema_message, 'DataFrame'),
                                      TypingInfoMismatchWarning)
                    else:
                        copied_schema = self._schema._get_subset_schema(list(self._dataframe.columns))
                        result.ww.init(schema=copied_schema)
                else:
                    # Confirm that the Schema is still valid on original DataFrame
                    # Important for inplace operations
                    invalid_schema_message = _get_invalid_schema_message(self._dataframe, self._schema)
                    if invalid_schema_message:
                        warnings.warn(TypingInfoMismatchWarning().get_warning_message(attr, invalid_schema_message, 'DataFrame'),
                                      TypingInfoMismatchWarning)
                        self._schema = None

                # Always return the results of the DataFrame operation whether or not Woodwork is initialized
                return result
            return wrapper
        # Directly return non-callable DataFrame attributes
        return dataframe_attr

    def _get_subset_df_with_schema(self, cols_to_include, use_dataframe_order=True):
        """Creates a new DataFrame from a list of column names with Woodwork initialized,
        retaining all typing information and maintaining the DataFrame's column order."""
        assert all([col_name in self._schema.columns for col_name in cols_to_include])

        if use_dataframe_order:
            cols_to_include = [col_name for col_name in self._dataframe.columns if col_name in cols_to_include]
        else:
            cols_to_include = [col_name for col_name in cols_to_include if col_name in self._dataframe.columns]

        new_schema = self._schema._get_subset_schema(cols_to_include)
        new_df = self._dataframe[cols_to_include]
        new_df.ww.init(schema=new_schema)

        return new_df

    def pop(self, column_name):
        """Return a Series with Woodwork typing information and remove it from the DataFrame.

        Args:
            column (str): Name of the column to pop.

        Returns:
            Series: Popped series with Woodwork initialized
        """
        if self._schema is None:
            _raise_init_error()
        if column_name not in self._dataframe.columns:
            raise LookupError(f'Column with name {column_name} not found in DataFrame')

        series = self._dataframe.pop(column_name)

        # Initialize Woodwork typing info for series
        col_schema = self._schema.columns[column_name]
        series.ww.init(**col_schema, use_standard_tags=self.use_standard_tags)

        # Update schema to not include popped column
        del self._schema.columns[column_name]

        return series

    def drop(self, columns):
        """Drop specified columns from a DataFrame.

        Args:
            columns (str or list[str]): Column name or names to drop. Must be present in the DataFrame.

        Returns:
            DataFrame: DataFrame with the specified columns removed, maintaining Woodwork typing information.

        Note:
            This method is used for removing columns only. To remove rows with ``drop``, go through the
            DataFrame directly and then reinitialize Woodwork with ``DataFrame.ww.init``
            instead of calling ``DataFrame.ww.drop``.
        """
        if self._schema is None:
            _raise_init_error()
        if not isinstance(columns, (list, set)):
            columns = [columns]

        not_present = [col for col in columns if col not in self._dataframe.columns]
        if not_present:
            raise ValueError(f'{not_present} not found in DataFrame')

        return self._get_subset_df_with_schema([col for col in self._dataframe.columns if col not in columns])

    def rename(self, columns):
        """Renames columns in a DataFrame, maintaining Woodwork typing information.

        Args:
            columns (dict[str -> str]): A dictionary mapping current column names to new column names.

        Returns:
            DataFrame: DataFrame with the specified columns renamed, maintaining Woodwork typing information.
        """
        if self._schema is None:
            _raise_init_error()
        new_schema = self._schema.rename(columns)
        new_df = self._dataframe.rename(columns=columns)

        new_df.ww.init(schema=new_schema)
        return new_df

    def mutual_information_dict(self, num_bins=10, nrows=None, include_index=False):
        """
        Calculates mutual information between all pairs of columns in the DataFrame that
        support mutual information. Logical Types that support mutual information are as
        follows:  Boolean, Categorical, CountryCode, Datetime, Double, Integer, Ordinal,
        SubRegionCode, and ZIPCode

        Args:
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical.
            nrows (int): The number of rows to sample for when determining mutual info.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for mutual information calculations.
                If False, the index column will not have mutual information calculated for it.
                Defaults to False.

        Returns:
            list(dict): A list containing dictionaries that have keys `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect dependency).
        """
        if self._schema is None:
            _raise_init_error()
        return _get_mutual_information_dict(self._dataframe, num_bins=num_bins, nrows=nrows, include_index=include_index)

    def mutual_information(self, num_bins=10, nrows=None, include_index=False):
        """Calculates mutual information between all pairs of columns in the DataFrame that
        support mutual information. Logical Types that support mutual information are as
        follows:  Boolean, Categorical, CountryCode, Datetime, Double, Integer, Ordinal,
        SubRegionCode, and ZIPCode

        Args:
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical.
            nrows (int): The number of rows to sample for when determining mutual info.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.
            include_index (bool): If True, the column specified as the index will be
                included as long as its LogicalType is valid for mutual information calculations.
                If False, the index column will not have mutual information calculated for it.
                Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing mutual information with columns `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect dependency).
        """
        mutual_info = self.mutual_information_dict(num_bins, nrows, include_index)
        return pd.DataFrame(mutual_info)

    def describe_dict(self, include=None):
        """Calculates statistics for data contained in the DataFrame.

        Args:
            include (list[str or LogicalType], optional): filter for what columns to include in the
                statistics returned. Can be a list of column names, semantic tags, logical types, or a list
                combining any of the three. It follows the most broad specification. Favors logical types
                then semantic tag then column name. If no matching columns are found, an empty DataFrame
                will be returned.

        Returns:
            dict[str -> dict]: A dictionary with a key for each column in the data or for each column
            matching the logical types, semantic tags or column names specified in ``include``, paired
            with a value containing a dictionary containing relevant statistics for that column.
        """
        if self._schema is None:
            _raise_init_error()
        return _get_describe_dict(self._dataframe, include=include)

    def describe(self, include=None):
        """Calculates statistics for data contained in the DataFrame.

        Args:
            include (list[str or LogicalType], optional): filter for what columns to include in the
                statistics returned. Can be a list of column names, semantic tags, logical types, or a list
                combining any of the three. It follows the most broad specification. Favors logical types
                then semantic tag then column name. If no matching columns are found, an empty DataFrame
                will be returned.

        Returns:
            pd.DataFrame: A Dataframe containing statistics for the data or the subset of the original
            DataFrame that contains the logical types, semantic tags, or column names specified
            in ``include``.
        """
        results = self.describe_dict(include=include)
        index_order = [
            'physical_type',
            'logical_type',
            'semantic_tags',
            'count',
            'nunique',
            'nan_count',
            'mean',
            'mode',
            'std',
            'min',
            'first_quartile',
            'second_quartile',
            'third_quartile',
            'max',
            'num_true',
            'num_false',
        ]
        return pd.DataFrame(results).reindex(index_order)

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
        if self._schema is None:
            _raise_init_error()
        return _get_value_counts(self._dataframe, ascending, top_n, dropna)


def _validate_accessor_params(dataframe, index, make_index, time_index, logical_types, schema):
    _check_unique_column_names(dataframe)
    if schema is not None:
        _check_schema(dataframe, schema)
    else:
        # We ignore these parameters if a schema is passed
        if index is not None or make_index:
            _check_index(dataframe, index, make_index)
        if logical_types:
            _check_logical_types(dataframe.columns, logical_types)
        if time_index is not None:
            datetime_format = None
            logical_type = None
            if logical_types is not None and time_index in logical_types:
                logical_type = logical_types[time_index]
                if _get_ltype_class(logical_types[time_index]) == Datetime:
                    datetime_format = logical_types[time_index].datetime_format

            _check_time_index(dataframe, time_index, datetime_format=datetime_format, logical_type=logical_type)


def _check_unique_column_names(dataframe):
    if not dataframe.columns.is_unique:
        raise IndexError('Dataframe cannot contain duplicate columns names')


def _check_index(dataframe, index, make_index=False):
    if not make_index and index not in dataframe.columns:
        # User specifies an index that is not in the dataframe, without setting make_index to True
        raise LookupError(f'Specified index column `{index}` not found in dataframe. '
                          'To create a new index column, set make_index to True.')
    if index is not None and not make_index and isinstance(dataframe, pd.DataFrame) and not dataframe[index].is_unique:
        # User specifies an index that is in the dataframe but not unique
        # Does not check for Dask as Dask does not support is_unique
        raise IndexError('Index column must be unique')
    if make_index and index is not None and index in dataframe.columns:
        # User sets make_index to True, but supplies an index name that matches a column already present
        raise IndexError('When setting make_index to True, '
                         'the name specified for index cannot match an existing column name')
    if make_index and index is None:
        # User sets make_index to True, but does not supply a name for the index
        raise IndexError('When setting make_index to True, '
                         'the name for the new index must be specified in the index parameter')


def _check_time_index(dataframe, time_index, datetime_format=None, logical_type=None):
    if time_index not in dataframe.columns:
        raise LookupError(f'Specified time index column `{time_index}` not found in dataframe')
    if not (_is_numeric_series(dataframe[time_index], logical_type) or
            col_is_datetime(dataframe[time_index], datetime_format=datetime_format)):
        raise TypeError('Time index column must contain datetime or numeric values')


def _check_logical_types(dataframe_columns, logical_types):
    if not isinstance(logical_types, dict):
        raise TypeError('logical_types must be a dictionary')
    cols_not_found = set(logical_types.keys()).difference(set(dataframe_columns))
    if cols_not_found:
        raise LookupError('logical_types contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')


def _check_schema(dataframe, schema):
    if not isinstance(schema, Schema):
        raise TypeError('Provided schema must be a Woodwork.Schema object.')
    invalid_schema_message = _get_invalid_schema_message(dataframe, schema)
    if invalid_schema_message:
        raise ValueError(f'Woodwork typing information is not valid for this DataFrame: {invalid_schema_message}')


def _make_index(dataframe, index):
    if dd and isinstance(dataframe, dd.DataFrame):
        dataframe[index] = 1
        dataframe[index] = dataframe[index].cumsum() - 1
    elif ks and isinstance(dataframe, ks.DataFrame):
        raise TypeError('Cannot make index on a Koalas DataFrame.')
    else:
        dataframe.insert(0, index, range(len(dataframe)))


def _get_invalid_schema_message(dataframe, schema):
    dataframe_cols = set(dataframe.columns)
    schema_cols = set(schema.columns.keys())

    df_cols_not_in_schema = dataframe_cols - schema_cols
    if df_cols_not_in_schema:
        return f'The following columns in the DataFrame were missing from the typing information: '\
            f'{df_cols_not_in_schema}'
    schema_cols_not_in_df = schema_cols - dataframe_cols
    if schema_cols_not_in_df:
        return f'The following columns in the typing information were missing from the DataFrame: '\
            f'{schema_cols_not_in_df}'
    for name in dataframe.columns:
        df_dtype = dataframe[name].dtype
        valid_dtype = _get_valid_dtype(type(dataframe[name]), schema.logical_types[name])
        if str(df_dtype) != valid_dtype:
            return f'dtype mismatch for column {name} between DataFrame dtype, '\
                f'{df_dtype}, and {schema.logical_types[name]} dtype, {valid_dtype}'
    if schema.index is not None and isinstance(dataframe, pd.DataFrame):
        # Index validation not performed for Dask/Koalas
        if not all(dataframe.index == dataframe[schema.index]):
            return 'Index mismatch between DataFrame and typing information'
        elif not dataframe[schema.index].is_unique:
            return 'Index column is not unique'


def _raise_init_error():
    raise AttributeError("Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init")


@pd.api.extensions.register_dataframe_accessor('ww')
class PandasTableAccessor(WoodworkTableAccessor):
    pass


if dd:
    @dd.extensions.register_dataframe_accessor('ww')
    class DaskTableAccessor(WoodworkTableAccessor):
        pass


if ks:
    from databricks.koalas.extensions import register_dataframe_accessor

    @register_dataframe_accessor('ww')
    class KoalasTableAccessor(WoodworkTableAccessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not ks.get_option('compute.ops_on_diff_frames'):
                ks.set_option('compute.ops_on_diff_frames', True)
