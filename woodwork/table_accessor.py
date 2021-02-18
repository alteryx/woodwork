import warnings

import pandas as pd

from woodwork.accessor_utils import _update_column_dtype
from woodwork.exceptions import (
    ParametersIgnoredWarning,
    TypingInfoMismatchWarning
)
from woodwork.logical_types import Datetime
from woodwork.schema import Schema
from woodwork.statistics_utils import (
    _get_describe_dict,
    _get_mutual_information_dict
)
from woodwork.type_sys.utils import (
    _get_ltype_class,
    _is_numeric_series,
    col_is_datetime
)
from woodwork.utils import _get_column_logical_type, import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


@pd.api.extensions.register_dataframe_accessor('ww')
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
        '''Initializes Woodwork typing information for a DataFrame.

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
        '''
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

    def __getattr__(self, attr):
        '''
            If the method is present on the Accessor, uses that method.
            If the method is present on Schema, uses that method.
            If the method is present on DataFrame, uses that method.
        '''
        if self._schema is None:
            raise AttributeError("Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init")
        if hasattr(self._schema, attr):
            return self._make_schema_call(attr)
        if hasattr(self._dataframe, attr):
            return self._make_dataframe_call(attr)
        else:
            raise AttributeError(f"Woodwork has no attribute '{attr}'")

    def __repr__(self):
        return repr(self._schema)

    @property
    def schema(self):
        ''' A copy of the Woodwork typing information for the DataFrame.
        '''
        if self._schema:
            return self._schema._get_subset_schema(list(self.columns.keys()))

    def select(self, include):
        """Create a DataFrame with Woodowork typing information initialized
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
        cols_to_include = self._schema._filter_cols(include)
        return self._get_subset_df_with_schema(cols_to_include)

    def _sort_columns(self, already_sorted):
        if dd and isinstance(self._dataframe, dd.DataFrame) or (ks and isinstance(self._dataframe, ks.DataFrame)):
            already_sorted = True  # Skip sorting for Dask and Koalas input
        if not already_sorted:
            sort_cols = [self._schema.time_index, self._schema.index]
            if self._schema.index is None:
                sort_cols = [self._schema.time_index]
            self._dataframe.sort_values(sort_cols, inplace=True)

    def _set_underlying_index(self):
        '''Sets the index of the underlying DataFrame.
        If there is an index specified for the Schema, will be set to that index.
        If no index is specified and the DataFrame's index isn't a RangeIndex, will reset the DataFrame's index,
        meaning that the index will be a pd.RangeIndex starting from zero.
        '''
        if isinstance(self._dataframe, pd.DataFrame):
            if self._schema.index is not None:
                self._dataframe.set_index(self._schema.index, drop=False, inplace=True)
                # Drop index name to not overlap with the original column
                self._dataframe.index.name = None
            # Only reset the index if the index isn't a RangeIndex
            elif not isinstance(self._dataframe.index, pd.RangeIndex):
                self._dataframe.reset_index(drop=True, inplace=True)

    def _make_schema_call(self, attr):
        '''
        Forwards the requested attribute onto the schema object.
        Results are that of the Woodwork.Schema class.
        '''
        schema_attr = getattr(self._schema, attr)

        if callable(schema_attr):
            def wrapper(*args, **kwargs):
                return schema_attr(*args, **kwargs)
            return wrapper
        return schema_attr

    def _make_dataframe_call(self, attr):
        '''
        Forwards the requested attribute onto the dataframe object.
        Intercepts return value, attempting to initialize Woodwork with the current schema
        when a new DataFrame is returned.
        Confirms schema is still valid for the original DataFrame.
        '''
        dataframe_attr = getattr(self._dataframe, attr)

        if callable(dataframe_attr):
            def wrapper(*args, **kwargs):
                # Make DataFrame call and intercept the result
                result = dataframe_attr(*args, **kwargs)

                # Try to initialize Woodwork with the existing Schema
                if isinstance(result, pd.DataFrame):
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

    def _get_subset_df_with_schema(self, cols_to_include):
        '''
        Creates a new DataFrame from a list of column names with Woodwork initialized,
        retaining all typing information and maintaining the DataFrame's column order.
        '''
        assert all([col_name in self._schema.columns for col_name in cols_to_include])
        cols_to_include = [col_name for col_name in self._dataframe.columns if col_name in cols_to_include]

        new_schema = self._schema._get_subset_schema(cols_to_include)

        new_df = self._dataframe[cols_to_include]
        new_df.ww.init(schema=new_schema)

        return new_df

    def mutual_information_dict(self, num_bins=10, nrows=None):
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

        Returns:
            list(dict): A list containing dictionaries that have keys `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect dependency).
        """
        return _get_mutual_information_dict(self._dataframe, num_bins=num_bins, nrows=nrows)

    def mutual_information(self, num_bins=10, nrows=None):
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

        Returns:
            pd.DataFrame: A DataFrame containing mutual information with columns `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect dependency).
        """
        mutual_info = self.mutual_information_dict(num_bins, nrows)
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
        results = _get_describe_dict(self._dataframe, include=include)
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
        schema_dtype = schema.logical_types[name].pandas_dtype
        if df_dtype != schema_dtype:
            return f'dtype mismatch for column {name} between DataFrame dtype, '\
                f'{df_dtype}, and {schema.logical_types[name]} dtype, {schema_dtype}'
    if schema.index is not None:
        if not all(dataframe.index == dataframe[schema.index]):
            return 'Index mismatch between DataFrame and typing information'
        elif not dataframe[schema.index].is_unique:
            return 'Index column is not unique'
