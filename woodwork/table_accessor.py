import warnings

import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

from woodwork.exceptions import TypingInfoMismatchWarning
from woodwork.logical_types import Datetime, Double, LatLong, Ordinal
from woodwork.schema import Schema
from woodwork.schema_column import (
    _is_col_boolean,
    _is_col_categorical,
    _is_col_datetime,
    _is_col_numeric
)
from woodwork.type_sys.utils import (
    _get_ltype_class,
    _is_numeric_series,
    col_is_datetime
)
from woodwork.utils import (
    _get_column_logical_type,
    _get_mode,
    _reformat_to_latlong,
    get_valid_mi_types,
    import_or_none
)

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')
if ks:
    ks.set_option('compute.ops_on_diff_frames', True)


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
                Any other arguments provided will be ignored.
        '''
        _validate_accessor_params(self._dataframe, index, make_index, time_index, logical_types, schema)
        if schema is not None:
            self._schema = schema
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

                updated_series = _update_column_dtype(series, logical_type, name)
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
        return self._schema

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
                        warnings.warn(TypingInfoMismatchWarning().get_warning_message(attr, invalid_schema_message),
                                      TypingInfoMismatchWarning)
                    else:
                        result.ww.init(schema=self._schema)
                else:
                    # Confirm that the Schema is still valid on original DataFrame
                    # Important for inplace operations
                    invalid_schema_message = _get_invalid_schema_message(self._dataframe, self._schema)
                    if invalid_schema_message:
                        warnings.warn(TypingInfoMismatchWarning().get_warning_message(attr, invalid_schema_message),
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

    def _replace_nans_for_mutual_info(self, data):
        """
        Replace NaN values in the dataframe so that mutual information can be calculated

        Args:
            data (pd.DataFrame): dataframe to use for calculating mutual information

        Returns:
            pd.DataFrame: data with nans replaced with either mean or mode

        """
        # replace or remove null values
        for column_name in data.columns[data.isnull().any()]:
            column = self._schema.columns[column_name]
            series = data[column_name]

            if _is_col_numeric(column) or _is_col_datetime(column):
                mean = series.mean()
                if isinstance(mean, float) and not _get_ltype_class(column['logical_type']) == Double:
                    data[column_name] = series.astype('float')
                data[column_name] = series.fillna(mean)
            elif _is_col_categorical(column) or _is_col_boolean(column):
                mode = _get_mode(series)
                data[column_name] = series.fillna(mode)
        return data

    def _make_categorical_for_mutual_info(self, data, num_bins):
        """Transforms dataframe columns into numeric categories so that
        mutual information can be calculated

        Args:
            data (pd.DataFrame): dataframe to use for caculating mutual information
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical.

        Returns:
            data (pd.DataFrame): Transformed data
        """

        col_names = data.columns.to_list()
        for col_name in col_names:
            column = self._schema.columns[col_name]
            if _is_col_numeric(column):
                # bin numeric features to make categories
                data[col_name] = pd.qcut(data[col_name], num_bins, duplicates="drop")
            # Convert Datetimes to total seconds - an integer - and bin
            if _is_col_datetime(column):
                data[col_name] = pd.qcut(data[col_name].astype('int64'), num_bins, duplicates="drop")
            # convert categories to integers
            new_col = data[col_name]
            if str(new_col.dtype) != 'category':
                new_col = new_col.astype('category')
            data[col_name] = new_col.cat.codes
        return data

    def mutual_information_dict(self, num_bins=10, nrows=None):
        """
        Calculates mutual information between all pairs of columns in the DataTable that
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
        valid_types = get_valid_mi_types()
        valid_columns = [col['name'] for col in self._schema.columns.values() if (
            col['name'] != self._schema.index and _get_ltype_class(col['logical_type']) in valid_types)]

        data = self._dataframe[valid_columns]
        if dd and isinstance(data, dd.DataFrame):
            data = data.compute()
        if ks and isinstance(self._dataframe, ks.DataFrame):
            data = data.to_pandas()

        # cut off data if necessary
        if nrows is not None and nrows < data.shape[0]:
            data = data.sample(nrows)

        # remove fully null columns
        not_null_cols = data.columns[data.notnull().any()]
        if set(not_null_cols) != set(valid_columns):
            data = data.loc[:, not_null_cols]
        # remove columns that are unique
        not_unique_cols = [col for col in data.columns if not data[col].is_unique]
        if set(not_unique_cols) != set(valid_columns):
            data = data.loc[:, not_unique_cols]

        data = self._replace_nans_for_mutual_info(data)
        data = self._make_categorical_for_mutual_info(data, num_bins)

        # calculate mutual info for all pairs of columns
        mutual_info = []
        col_names = data.columns.to_list()
        for i, a_col in enumerate(col_names):
            for j in range(i, len(col_names)):
                b_col = col_names[j]
                if a_col == b_col:
                    # Ignore because the mutual info for a column with itself will always be 1
                    continue
                else:
                    mi_score = normalized_mutual_info_score(data[a_col], data[b_col])
                    mutual_info.append(
                        {"column_1": a_col, "column_2": b_col, "mutual_info": mi_score}
                    )
        mutual_info.sort(key=lambda mi: mi['mutual_info'], reverse=True)
        return mutual_info

    def mutual_information(self, num_bins=10, nrows=None):
        """
        Calculates mutual information between all pairs of columns in the DataTable that
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
            pd.DataFrame: A Dataframe containing mutual information with columns `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect dependency).
        """
        mutual_info = self.mutual_information_dict(num_bins, nrows)
        return pd.DataFrame(mutual_info)


def _validate_accessor_params(dataframe, index, make_index, time_index, logical_types, schema):
    _check_unique_column_names(dataframe)
    if schema is not None:
        _check_schema(dataframe, schema)
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


def _update_column_dtype(series, logical_type, name):
    """Update the dtype of the underlying series to match the dtype corresponding
    to the LogicalType for the column."""
    if isinstance(logical_type, Ordinal):
        logical_type._validate_data(series)
    elif _get_ltype_class(logical_type) == LatLong:
        # Reformat LatLong columns to be a length two tuple (or list for Koalas) of floats
        if dd and isinstance(series, dd.Series):
            name = series.name
            meta = (series, tuple([float, float]))
            series = series.apply(_reformat_to_latlong, meta=meta)
            series.name = name
        elif ks and isinstance(series, ks.Series):
            formatted_series = series.to_pandas().apply(_reformat_to_latlong, use_list=True)
            series = ks.from_pandas(formatted_series)
        else:
            series = series.apply(_reformat_to_latlong)

    if logical_type.pandas_dtype != str(series.dtype):
        # Update the underlying series
        try:
            if _get_ltype_class(logical_type) == Datetime:
                if dd and isinstance(series, dd.Series):
                    name = series.name
                    series = dd.to_datetime(series, format=logical_type.datetime_format)
                    series.name = name
                elif ks and isinstance(series, ks.Series):
                    series = ks.Series(ks.to_datetime(series.to_numpy(),
                                                      format=logical_type.datetime_format),
                                       name=series.name)
                else:
                    series = pd.to_datetime(series, format=logical_type.datetime_format)
            else:
                if ks and isinstance(series, ks.Series) and logical_type.backup_dtype:
                    new_dtype = logical_type.backup_dtype
                else:
                    new_dtype = logical_type.pandas_dtype
                series = series.astype(new_dtype)
        except (TypeError, ValueError):
            error_msg = f'Error converting datatype for column {name} from type {str(series.dtype)} ' \
                f'to type {logical_type.pandas_dtype}. Please confirm the underlying data is consistent with ' \
                f'logical type {logical_type}.'
            raise TypeError(error_msg)
    return series


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
