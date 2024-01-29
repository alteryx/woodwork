import ast
import importlib
import re
from datetime import datetime
from inspect import isclass
from mimetypes import add_type, guess_type
from timeit import default_timer as timer

import numpy as np
import pandas as pd

import woodwork as ww
from woodwork.exceptions import TypeValidationError

# Dictionary mapping formats/content types to the appropriate pandas read function

type_to_read_func_map = {
    "csv": pd.read_csv,
    "text/csv": pd.read_csv,
    "parquet": pd.read_parquet,
    "application/parquet": pd.read_parquet,
    "arrow": pd.read_feather,
    "application/arrow": pd.read_feather,
    "feather": pd.read_feather,
    "application/feather": pd.read_feather,
    "orc": pd.read_orc,
    "application/orc": pd.read_orc,
}

PYARROW_ERR_MSG = (
    "The pyarrow library is required to read from parquet/arrow/feather files.\n"
    "Install via pip:\n"
    "    pip install 'pyarrow>=3.0.0'\n"
    "Install via conda:\n"
    "    conda install 'pyarrow>=3.0.0'"
)

# Add new mimetypes
add_type("application/parquet", ".parquet")
add_type("application/arrow", ".arrow")
add_type("application/feather", ".feather")
add_type("application/orc", ".orc")


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
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def _convert_input_to_set(semantic_tags, error_language="semantic_tags", validate=True):
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


def read_file(
    filepath=None,
    content_type=None,
    name=None,
    index=None,
    time_index=None,
    semantic_tags=None,
    logical_types=None,
    use_standard_tags=True,
    column_origins=None,
    replace_nan=False,
    validate=True,
    **kwargs,
):
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
        replace_nan (bool, optional): Whether to replace empty string values and string representations of
            NaN values ("nan", "<NA>") with np.nan or pd.NA values based on column dtype. Defaults to False.
        validate (bool, optional): Whether parameter and data validation should occur. Defaults to True. Warning:
                Should be set to False only when parameters and data are known to be valid.
                Any errors resulting from skipping validation with invalid inputs may not be easily understood.
        **kwargs: Additional keyword arguments to pass to the underlying pandas read file function. For more
            information on available keywords refer to the pandas documentation.

    Returns:
        pd.DataFrame: DataFrame created from the specified file with Woodwork typing information initialized.
    """
    from woodwork.logical_types import _replace_nans

    if content_type is None:
        inferred_type, _ = guess_type(filepath)
        if inferred_type is None:
            raise RuntimeError(
                "Content type could not be inferred. Please specify content_type and try again.",
            )
        content_type = inferred_type

    if content_type not in type_to_read_func_map:
        raise RuntimeError(
            "Reading from content type {} is not currently supported".format(
                content_type,
            ),
        )

    pyarrow_types = [
        "parquet",
        "application/parquet",
        "arrow",
        "application/arrow",
        "feather",
        "application/feather",
        "orc",
        "application/orc",
    ]
    if content_type in pyarrow_types:
        import_or_raise("pyarrow", PYARROW_ERR_MSG)
        if content_type in ["parquet", "application/parquet"]:
            kwargs["engine"] = "pyarrow"

    dataframe = type_to_read_func_map[content_type](filepath, **kwargs)

    if replace_nan:
        dataframe = dataframe.apply(_replace_nans)

    dataframe.ww.init(
        name=name,
        index=index,
        time_index=time_index,
        semantic_tags=semantic_tags,
        logical_types=logical_types,
        use_standard_tags=use_standard_tags,
        column_origins=column_origins,
        validate=validate,
    )
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
    return "http" in string


def _reformat_to_latlong(latlong, is_spark=False):
    """
    Accepts 2-tuple like values, or a single NaN like value.
    NaN like values are replaced with np.nan.
    """
    if isinstance(latlong, str):
        latlong = _parse_latlong(latlong) or latlong

    if isinstance(latlong, (list, tuple)):
        if len(latlong) != 2:
            raise TypeValidationError(
                f"LatLong values must have exactly two values. {latlong} does not have two values.",
            )

        latitude, longitude = latlong

        try:
            latitude = _coerce_to_float(latitude)
            longitude = _coerce_to_float(longitude)
        except ValueError:
            raise TypeValidationError(
                f"LatLong values must be in decimal degrees. {latlong} does not have latitude or longitude values that can be converted to a float.",
            )

        latlong = (latitude, longitude)
        if is_spark:
            latlong = list(latlong)
        return latlong

    if _is_nan(latlong):
        return np.nan

    raise TypeValidationError(
        f"""LatLong value is not properly formatted. Value must be one of the following:
- A 2-tuple or list of 2 values representing decimal latitude or longitude values (NaN values are allowed).
- A single NaN value.
- A string representation of the above.

{latlong} does not fit the criteria.""",
    )


def _coerce_to_float(val):
    """Attempts to convert a value to a float, propagating null values."""
    if _is_nan(val):
        return np.nan

    try:
        return float(val)
    except (ValueError, TypeError):
        raise ValueError(
            f"The value represented by {val} cannot be converted to a float.",
        )


def _is_valid_latlong_series(series):
    """Returns True if all elements in the series contain properly formatted LatLong values,
    otherwise returns False"""
    if ww.accessor_utils._is_dask_series(series):
        series = series.get_partition(0).compute()
    if ww.accessor_utils._is_spark_series(series):
        series = series.to_pandas()
        is_spark = True
    else:
        is_spark = False
    if series.apply(_is_valid_latlong_value, args=(is_spark,)).all():
        return True
    return False


def _is_valid_latlong_value(val, is_spark=False):
    """Returns True if the value provided is a properly formatted LatLong value for a
    pandas, Dask or Spark Series, otherwise returns False."""
    if isinstance(val, (list, tuple)):
        if len(val) != 2:
            return False

        if not isinstance(val, list if is_spark else tuple):
            return False

        latitude, longitude = val
        lat_null, long_null = map(pd.isnull, val)
        is_valid = isinstance(latitude, float) or lat_null
        is_valid &= isinstance(longitude, float) or long_null
        return is_valid

    if isinstance(val, float):
        return np.isnan(val)

    if isinstance(val, str):
        val = _parse_latlong(val)
        if val is None:
            return False
        else:
            return _is_valid_latlong_value(val)

    if is_spark and val is None:
        return True

    return False


def _is_nan(value):
    """This function checks if string values are common NaN values.
    Lists are not counted as NaN, and all other values are passed to pd.isnull
    """
    if isinstance(value, str):
        return value in ww.config.get_option("nan_values")
    if isinstance(value, list):
        return False
    return pd.isnull(value)


def _is_latlong_nan(value):
    """Checks if a LatLong value is NaN"""
    if isinstance(value, (tuple, list)):
        return all([_is_nan(x) for x in value])

    return _is_nan(value)


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
        if "category" in ltype.standard_tags:
            valid_types.append(ltype)
        elif "numeric" in ltype.standard_tags:
            valid_types.append(ltype)
        elif (
            ltype == ww.logical_types.Datetime
            or ltype == ww.logical_types.Boolean
            or ltype == ww.logical_types.BooleanNullable
        ):
            valid_types.append(ltype)

    return valid_types


def get_valid_pearson_types():
    """
    Generate a list of LogicalTypes that are valid for calculating Pearson correlation. Note that
    index columns are not valid for calculating dependence, but their types may be
    returned by this function.

    Args:
        None

    Returns:
        list(LogicalType): A list of the LogicalTypes that can be use to calculate Pearson correlation
    """
    valid_types = []
    for ltype in ww.type_system.registered_types:
        if "numeric" in ltype.standard_tags:
            valid_types.append(ltype)
        elif ltype == ww.logical_types.Datetime:
            valid_types.append(ltype)

    return valid_types


def get_valid_spearman_types():
    """
    Generate a list of LogicalTypes that are valid for calculating Spearman correlation. Note that
    index columns are not valid for calculating dependence, but their types may be
    returned by this function.

    Args:
        None

    Returns:
        list(LogicalType): A list of the LogicalTypes that can be use to calculate Spearman correlation
    """
    valid_types = []
    for ltype in ww.type_system.registered_types:
        if "numeric" in ltype.standard_tags:
            valid_types.append(ltype)
        elif ltype == ww.logical_types.Datetime or ltype == ww.logical_types.Ordinal:
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
        raise ValueError("No objects to concatenate")

    table_name = ""

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
                raise ValueError(
                    f"Cannot resolve overlapping keys in table metadata: {overlapping_keys}",
                )

            table_metadata = {**obj.ww.metadata, **table_metadata}

            # Combine table names
            if obj.ww.name is not None:
                if table_name:
                    table_name += "_"
                table_name += str(obj.ww.name)

            # Cannot have multiple tables with indexes or time indexes set
            if obj.ww.index is not None:
                if index is None:
                    index = obj.ww.index
                else:
                    raise IndexError(
                        "Cannot set the Woodwork index of multiple input objects. "
                        "Please remove the index columns from all but one table.",
                    )
            if obj.ww.time_index is not None:
                if time_index is None:
                    time_index = obj.ww.time_index
                else:
                    raise IndexError(
                        "Cannot set the Woodwork time index of multiple input objects. "
                        "Please remove the time index columns from all but one table.",
                    )

            ww_columns = obj.ww.schema.columns
        elif isinstance(obj.ww.schema, ww.column_schema.ColumnSchema):
            ww_columns = {obj.name: obj.ww.schema}

        # Compile the typing information per column
        for name, col_schema in ww_columns.items():
            if name in col_names_seen:
                raise ValueError(
                    f"Duplicate column '{name}' has been found in more than one input object. "
                    "Please remove duplicate columns from all but one table.",
                )
            logical_types[name] = col_schema.logical_type
            semantic_tags[name] = col_schema.semantic_tags - {"time_index"} - {"index"}
            col_metadata[name] = col_schema.metadata
            col_descriptions[name] = col_schema.description
            col_origins[name] = col_schema.origin
            use_standard_tags[name] = col_schema.use_standard_tags

            col_names_seen.add(name)

    # Perform concatenation with the correct library
    obj = objs[0]
    dd = import_or_none("dask.dataframe")
    ps = import_or_none("pyspark.pandas")

    lib = pd
    if ww.accessor_utils._is_spark_dataframe(obj) or ww.accessor_utils._is_spark_series(
        obj,
    ):
        lib = ps
    elif ww.accessor_utils._is_dask_dataframe(obj) or ww.accessor_utils._is_dask_series(
        obj,
    ):
        lib = dd

    combined_df = lib.concat(objs, axis=1, join="outer")

    # The lib.concat breaks the woodwork schema for dataframes with different shapes
    # or mismatched indices.
    mask = combined_df.isnull().any()
    null_cols = mask[mask].index
    if not ww.accessor_utils._is_dask_dataframe(combined_df):
        null_cols = null_cols.to_numpy()
    else:
        null_cols = list(null_cols)
    for null_col in null_cols:
        if null_col in logical_types and isinstance(
            logical_types[null_col],
            ww.logical_types.Integer,
        ):
            logical_types.pop(null_col)

    # Initialize Woodwork with all of the typing information from the input objs
    # performing type inference on any columns that did not already have Woodwork initialized
    combined_df.ww.init(
        name=table_name or None,
        index=index,
        time_index=time_index,
        logical_types=logical_types,
        semantic_tags=semantic_tags,
        table_metadata=table_metadata or None,
        column_metadata=col_metadata,
        column_descriptions=col_descriptions,
        column_origins=col_origins,
        use_standard_tags=use_standard_tags,
        validate=validate_schema,
    )
    return combined_df


class CallbackCaller:
    """
    Helper class for updating progress of a function and making a call to the progress callback
    function, if provided. Adds the progress increment to the current progress.

    If provided, the callback function should accept the following parameters:
        - update (int): change in progress since last call
        - progress (int): the progress so far in the calculations
        - total (int): the total number of calculations to do
        - unit (str): unit of measurement for progress/total
        - time_elapsed (float): total time in seconds elapsed since start of call

    """

    def __init__(self, callback, unit, total, start_time=None, start_progress=0):
        """
        Args:
            callback (func): callback method to call
            unit (str): unit of measurement for progress/total
            total (int): the total number of calculations to do
            start_time (datetime): when time started for the callback.  Defaults
                to when the class instance is created
            start_progress (int): starting progress for the callback.  Defaults to 0.
        """
        if start_time is not None:
            self.start_time = start_time
        else:
            self.start_time = timer()
        self.callback = callback
        self.unit = unit
        self.current_progress = start_progress
        self.total = total

    def update(self, progress_increment):
        """
        Args:
            progress_increment (int): change in progress since the last call
        """
        if self.callback is not None:
            elapsed_time = timer() - self.start_time
            new_progress = self.current_progress + progress_increment
            self.callback(
                progress_increment,
                new_progress,
                self.total,
                self.unit,
                elapsed_time,
            )
            self.current_progress = new_progress


def _infer_datetime_format(dates, n=100):
    """Helper function to infer the datetime format of the first n non-null rows of a series
    Args:
        dates (Series): Series of string or datetime string to guess the format of
        n (int): the maximum number of nonnull rows to sample from the series
    """
    dates_no_null = dates.dropna()

    ps = import_or_none("pyspark.pandas")
    dd = import_or_none("dask.dataframe")
    if ps and isinstance(dates_no_null, ps.series.Series):
        dates_no_null = dates_no_null.to_pandas()
    if dd and isinstance(dates_no_null, dd.Series):
        dates_no_null = dates_no_null.compute()

    random_n = dates_no_null.sample(min(n, len(dates_no_null)), random_state=42)

    if len(random_n) == 0:
        return None
    try:
        fmts = random_n.astype(str).map(pd.core.tools.datetimes.guess_datetime_format)
        mode_fmt = fmts.mode().loc[0]  # select first most common format
    except KeyError:
        check_for_other_formats = [
            "%y/%m/%d",
            "%m/%d/%y",
            "%d/%m/%y",
            "%y/%d/%m",
            "%d/%y/%m",
            "%m/%y/%d",
            "%d/%Y/%m",
            "%m/%Y/%d",
        ]
        dash_formats = []
        for format_ in check_for_other_formats:
            dash_formats.append(format_.replace("/", "-"))
        dot_formats = []
        for format_ in check_for_other_formats:
            dot_formats.append(format_.replace("/", "."))
        datetime_only_formats = check_for_other_formats + dash_formats + dot_formats

        time_stamp_formats = []
        for format_ in datetime_only_formats:
            time_stamp_formats.append(format_ + " %H:%M:%S")
        time_stamp_formats_with_timezone = []
        for format_ in datetime_only_formats:
            time_stamp_formats_with_timezone.append(format_ + " %H:%M:%S%z")

        check_for_other_formats = (
            datetime_only_formats
            + time_stamp_formats
            + time_stamp_formats_with_timezone
        )
        mode_fmt = None
        for format_ in check_for_other_formats:
            try:
                random_n.map(lambda x: datetime.strptime(x, format_))
                return format_
            except ValueError:  # Format doesn't match
                continue
            except TypeError:  # TimeStamp found instead of string
                break
    return mode_fmt


def _parse_latlong(latlong):
    nan_values_strs = [
        x
        for x in ww.config.get_option("nan_values")
        if isinstance(x, str) and len(x) and x != " "
    ]
    nan_values = "|".join(nan_values_strs)

    latlong = re.sub(nan_values, "None", latlong)

    try:
        return ast.literal_eval(latlong)
    except ValueError:
        pass
