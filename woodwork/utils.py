import ast
import importlib
import re
from inspect import isclass
from mimetypes import add_type, guess_type

import numpy as np
import pandas as pd

import woodwork as ww

# Dictionary mapping formats/content types to the appropriate pandas read function
type_to_read_func_map = {
    'csv': pd.read_csv,
    'text/csv': pd.read_csv,
    'parquet': pd.read_parquet,
    'application/parquet': pd.read_parquet,
    'arrow': pd.read_feather,
    'application/arrow': pd.read_feather,
    'feather': pd.read_feather,
    'application/feather': pd.read_feather,
    'orc': pd.read_orc,
    'application/orc': pd.read_orc,
}

PYARROW_ERR_MSG = (
    "The pyarrow library is required to read from parquet/arrow/feather files.\n"
    "Install via pip:\n"
    "    pip install 'pyarrow>=3.0.0'\n"
    "Install via conda:\n"
    "    conda install 'pyarrow>=3.0.0'"
)

# Add new mimetypes
add_type('application/parquet', '.parquet')
add_type('application/arrow', '.arrow')
add_type('application/feather', '.feather')
add_type('application/orc', '.orc')


def import_or_none(library):
    """Attempts to import the requested library.

    Args:
        library (str): the name of the library
    Returns: the library if it is installed, else None
    """
    try:
        return importlib.import_module(library)
    except ImportError:
        return None


def camel_to_snake(s):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def _convert_input_to_set(semantic_tags, error_language='semantic_tags', validate=True):
    """Takes input as a single string, a list of strings, or a set of strings
        and returns a set with the supplied values. If no values are supplied,
        an empty set will be returned."""
    if not semantic_tags:
        return set()

    if validate:
        _validate_tags_input_type(semantic_tags, error_language)

    if isinstance(semantic_tags, str):
        return {semantic_tags}

    if isinstance(semantic_tags, list):
        semantic_tags = set(semantic_tags)

    if validate:
        _validate_string_tags(semantic_tags, error_language)

    return semantic_tags


def _validate_tags_input_type(semantic_tags, error_language):
    if type(semantic_tags) not in [list, set, str]:
        raise TypeError(f"{error_language} must be a string, set or list")


def _validate_string_tags(semantic_tags, error_language):
    if not all([isinstance(tag, str) for tag in semantic_tags]):
        raise TypeError(f"{error_language} must contain only strings")


def read_file(filepath=None,
              content_type=None,
              name=None,
              index=None,
              time_index=None,
              semantic_tags=None,
              logical_types=None,
              use_standard_tags=True,
              column_origins=None,
              validate=True,
              **kwargs):
    """Read data from the specified file and return a DataFrame with initialized Woodwork typing information.

        Note:
            As the engine `fastparquet` cannot handle nullable pandas dtypes, `pyarrow` will be used
            for reading from parquet and arrow.

    Args:
        filepath (str): A valid string path to the file to read
        content_type (str): Content type of file to read
        name (str, optional): Name used to identify the DataFrame.
        index (str, optional): Name of the index column.
        time_index (str, optional): Name of the time index column.
        semantic_tags (dict, optional): Dictionary mapping column names in the dataframe to the
            semantic tags for the column. The keys in the dictionary should be strings
            that correspond to columns in the underlying dataframe. There are two options for
            specifying the dictionary values:
            (str): If only one semantic tag is being set, a single string can be used as a value.
            (list[str] or set[str]): If multiple tags are being set, a list or set of strings can be
            used as the value.
            Semantic tags will be set to an empty set for any column not included in the
            dictionary.
        logical_types (dict[str -> LogicalType], optional): Dictionary mapping column names in
            the dataframe to the LogicalType for the column. LogicalTypes will be inferred
            for any columns not present in the dictionary.
        use_standard_tags (bool, optional): If True, will add standard semantic tags to columns based
            on the inferred or specified logical type for the column. Defaults to True.
        column_origins (str or dict[str -> str], optional): Origin of each column. If a string is supplied, it is
                used as the origin for all columns. A dictionary can be used to set origins for individual columns.
        validate (bool, optional): Whether parameter and data validation should occur. Defaults to True. Warning:
                Should be set to False only when parameters and data are known to be valid.
                Any errors resulting from skipping validation with invalid inputs may not be easily understood.
        **kwargs: Additional keyword arguments to pass to the underlying pandas read file function. For more
            information on available keywords refer to the pandas documentation.

    Returns:
        pd.DataFrame: DataFrame created from the specified file with Woodwork typing information initialized.
    """
    if content_type is None:
        inferred_type, _ = guess_type(filepath)
        if inferred_type is None:
            raise RuntimeError('Content type could not be inferred. Please specify content_type and try again.')
        content_type = inferred_type

    if content_type not in type_to_read_func_map:
        raise RuntimeError('Reading from content type {} is not currently supported'.format(content_type))

    pyarrow_types = ['parquet', 'application/parquet',
                     'arrow', 'application/arrow',
                     'feather', 'application/feather',
                     'orc', 'application/orc']
    if content_type in pyarrow_types:
        import_or_raise('pyarrow', PYARROW_ERR_MSG)
        if content_type in ['parquet', 'application/parquet']:
            kwargs['engine'] = 'pyarrow'

    dataframe = type_to_read_func_map[content_type](filepath, **kwargs)
    dataframe.ww.init(name=name,
                      index=index,
                      time_index=time_index,
                      semantic_tags=semantic_tags,
                      logical_types=logical_types,
                      use_standard_tags=use_standard_tags,
                      column_origins=column_origins,
                      validate=validate)
    return dataframe


def import_or_raise(library, error_msg):
    """Attempts to import the requested library.  If the import fails, raises an
    ImportError with the supplied error message.

    Args:
        library (str): the name of the library
        error_msg (str): error message to return if the import fails
    """
    try:
        return importlib.import_module(library)
    except ImportError:
        raise ImportError(error_msg)


def _is_s3(string):
    """Checks if the given string is a s3 path. Returns a boolean."""
    return "s3://" in string


def _is_url(string):
    """Checks if the given string is an url path. Returns a boolean."""
    return 'http' in string


def _reformat_to_latlong(latlong, use_list=False):
    """Reformats LatLong columns to be tuples of floats. Uses np.nan for null values."""
    if _is_null_latlong(latlong):
        return np.nan

    if isinstance(latlong, str):
        try:
            # Serialized latlong columns from csv or parquet will be strings, so null values will be
            # read as the string 'nan' in pandas and Dask and 'NaN' in Koalas
            # neither of which which is interpretable as a null value
            if 'nan' in latlong:
                latlong = latlong.replace('nan', 'None')
            if 'NaN' in latlong:
                latlong = latlong.replace('NaN', 'None')
            latlong = ast.literal_eval(latlong)
        except ValueError:
            pass

    if isinstance(latlong, (tuple, list)):
        if len(latlong) != 2:
            raise ValueError(f'LatLong values must have exactly two values. {latlong} does not have two values.')

        latitude, longitude = map(_to_latlong_float, latlong)

        # (np.nan, np.nan) should be counted as a single null value
        if pd.isnull(latitude) and pd.isnull(longitude):
            return np.nan

        if use_list:
            return [latitude, longitude]
        return (latitude, longitude)

    raise ValueError(f'LatLongs must either be a tuple, a list, or a string representation of a tuple. {latlong} does not fit the criteria.')


def _to_latlong_float(val):
    """Attempts to convert a value to a float, propagating null values."""
    if _is_null_latlong(val):
        return np.nan

    try:
        return float(val)
    except (ValueError, TypeError):
        raise ValueError(f'Latitude and Longitude values must be in decimal degrees. The latitude or longitude represented by {val} cannot be converted to a float.')


def _is_valid_latlong_series(series):
    """Returns True if all elements in the series contain properly formatted LatLong values,
    otherwise returns False"""
    if ww.accessor_utils._is_dask_series(series):
        series = series = series.get_partition(0).compute()
    if ww.accessor_utils._is_koalas_series(series):
        series = series.to_pandas()
        bracket_type = list
    else:
        bracket_type = tuple
    if series.apply(_is_valid_latlong_value, args=(bracket_type,)).all():
        return True
    return False


def _is_valid_latlong_value(val, bracket_type=tuple):
    """Returns True if the value provided is a properly formatted LatLong value for a
    pandas, Dask or Koalas Series, otherwise returns False."""
    if isinstance(val, bracket_type) and len(val) == 2:
        latitude, longitude = val
        if isinstance(latitude, float) and isinstance(longitude, float):
            if pd.isnull(latitude) and pd.isnull(longitude):
                return False
            return True
    elif isinstance(val, float) and pd.isnull(val):
        return True
    return False


def _is_null_latlong(val):
    if isinstance(val, str):
        return val == 'None' or val == 'nan' or val == 'NaN'

    # Since we can have list inputs here, pd.isnull will not have a relevant truth value for lists
    return not isinstance(val, list) and pd.isnull(val)


def get_valid_mi_types():
    """
    Generate a list of LogicalTypes that are valid for calculating mutual information. Note that
    index columns are not valid for calculating mutual information, but their types may be
    returned by this function.

    Args:
        None

    Returns:
        list(LogicalType): A list of the LogicalTypes that can be use to calculate mutual information
    """
    valid_types = []
    for ltype in ww.type_system.registered_types:
        if 'category' in ltype.standard_tags:
            valid_types.append(ltype)
        elif 'numeric' in ltype.standard_tags:
            valid_types.append(ltype)
        elif (ltype == ww.logical_types.Datetime or ltype == ww.logical_types.Boolean or
                ltype == ww.logical_types.BooleanNullable):
            valid_types.append(ltype)

    return valid_types


def _get_column_logical_type(series, logical_type, name):
    if logical_type:
        return _parse_logical_type(logical_type, name)
    else:
        return ww.type_system.infer_logical_type(series)


def _parse_logical_type(logical_type, name):
    if isinstance(logical_type, str):
        logical_type = ww.type_system.str_to_logical_type(logical_type)

    if isclass(logical_type):
        logical_type = logical_type()

    if type(logical_type) not in ww.type_system.registered_types:
        raise TypeError(f"Invalid logical type specified for '{name}'")

    return logical_type


def concat_columns(objs, validate_schema=True):
    """
    Concatenate Woodwork objects along the columns axis. There can only be one index and time index
    set across the objects passed in. As Woodwork does not allow duplicate column names,
    will not allow duplicate columns at concatenation.

    Args:
        objs (list[Series, DataFrame]): The Woodwork objects to be concatenated. If Woodwork
            is not initialized on any of the objects, type inference will be performed.
        validate_schema (bool, optional): Whether validation should be performed on the typing information
            for the concatenated DataFrame. Defaults to True.

    Returns:
        DataFrame: A Woodwork dataframe whose typing information is also a concatenation of the input dataframes.
    """
    if not objs:
        raise ValueError('No objects to concatenate')

    table_name = ''

    logical_types = {}
    semantic_tags = {}
    col_descriptions = {}
    col_origins = {}
    col_metadata = {}
    table_metadata = {}
    use_standard_tags = {}

    index = None
    time_index = None

    # Record the typing information for all the columns that have Woodwork schemas
    col_names_seen = set()
    for obj in objs:
        ww_columns = {}
        if isinstance(obj.ww.schema, ww.table_schema.TableSchema):
            # Raise error if there's overlap between table metadata
            overlapping_keys = obj.ww.metadata.keys() & table_metadata.keys()
            if overlapping_keys:
                raise ValueError(f'Cannot resolve overlapping keys in table metadata: {overlapping_keys}')

            table_metadata = {**obj.ww.metadata, **table_metadata}

            # Combine table names
            if obj.ww.name is not None:
                if table_name:
                    table_name += '_'
                table_name += str(obj.ww.name)

            # Cannot have multiple tables with indexes or time indexes set
            if obj.ww.index is not None:
                if index is None:
                    index = obj.ww.index
                else:
                    raise IndexError('Cannot set the Woodwork index of multiple input objects. '
                                     'Please remove the index columns from all but one table.')
            if obj.ww.time_index is not None:
                if time_index is None:
                    time_index = obj.ww.time_index
                else:
                    raise IndexError('Cannot set the Woodwork time index of multiple input objects. '
                                     'Please remove the time index columns from all but one table.')

            ww_columns = obj.ww.schema.columns
        elif isinstance(obj.ww.schema, ww.column_schema.ColumnSchema):
            ww_columns = {obj.name: obj.ww.schema}

        # Compile the typing information per column
        for name, col_schema in ww_columns.items():
            if name in col_names_seen:
                raise ValueError(f"Duplicate column '{name}' has been found in more than one input object. "
                                 "Please remove duplicate columns from all but one table.")
            logical_types[name] = col_schema.logical_type
            semantic_tags[name] = col_schema.semantic_tags - {'time_index'} - {'index'}
            col_metadata[name] = col_schema.metadata
            col_descriptions[name] = col_schema.description
            col_origins[name] = col_schema.origin
            use_standard_tags[name] = col_schema.use_standard_tags

            col_names_seen.add(name)

    # Perform concatenation with the correct library
    obj = objs[0]
    dd = import_or_none('dask.dataframe')
    ks = import_or_none('databricks.koalas')

    lib = pd
    if ww.accessor_utils._is_koalas_dataframe(obj) or ww.accessor_utils._is_koalas_series(obj):
        lib = ks
    elif ww.accessor_utils._is_dask_dataframe(obj) or ww.accessor_utils._is_dask_series(obj):
        lib = dd

    combined_df = lib.concat(objs, axis=1, join='outer')

    # Initialize Woodwork with all of the typing information from the input objs
    # performing type inference on any columns that did not already have Woodwork initialized
    combined_df.ww.init(name=table_name or None,
                        index=index,
                        time_index=time_index,
                        logical_types=logical_types,
                        semantic_tags=semantic_tags,
                        table_metadata=table_metadata or None,
                        column_metadata=col_metadata,
                        column_descriptions=col_descriptions,
                        column_origins=col_origins,
                        use_standard_tags=use_standard_tags,
                        validate=validate_schema)
    return combined_df


def _update_progress(start_time, current_time, progress_increment,
                     current_progress, total, unit, callback_function):
    """Helper function for updating progress of a function and making a call to the progress callback
    function, if provided. Adds the progress increment to the current progress amount and returns the
    updated progress amount.

    If provided, the callback function should accept the following parameters:
        - update (int): change in progress since last call
        - progress (int): the progress so far in the calculations
        - total (int): the total number of calculations to do
        - unit (str): unit of measurement for progress/total
        - time_elapsed (float): total time in seconds elapsed since start of call
    """
    if callback_function is not None:
        new_progress = current_progress + progress_increment
        elapsed_time = current_time - start_time
        callback_function(progress_increment, new_progress, total, unit, elapsed_time)

        return new_progress


def _infer_datetime_format(dates, n=100):
    """Helper function to infer the datetime format of the first n non-null rows of a series
    Args:
        dates (Series): Series of string or datetime string to guess the format of
        n (int): the maximum number of nonnull rows to sample from the series
    """
    first_n = dates.dropna().head(n)
    if len(first_n) == 0:
        return None
    try:
        fmts = first_n.map(pd.core.tools.datetimes.guess_datetime_format)
        mode_fmt = fmts.mode().loc[0]  # select first most common format
    except (TypeError, ValueError, IndexError, KeyError, NotImplementedError):
        mode_fmt = None
    return mode_fmt
