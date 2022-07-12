from functools import wraps

import numpy as np
import pandas as pd

from woodwork.exceptions import ColumnNotPresentInSchemaError, WoodworkNotInitError
from woodwork.utils import _get_column_logical_type, import_or_none

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


def init_series(
    series,
    logical_type=None,
    semantic_tags=None,
    use_standard_tags=True,
    null_invalid_values=False,
    description=None,
    origin=None,
    metadata=None,
):
    """Initializes Woodwork typing information for a series, numpy.ndarray or pd.api.extensions.
    ExtensionArray, returning a new Series. The dtype of the returned series will be converted
    to match the dtype associated with the LogicalType.

    Args:
        series (pd.Series, dd.Series, ps.Series, numpy.ndarray or pd.api.extensions.ExtensionArray):
            The original series from which to create the Woodwork initialized series.
        logical_type (LogicalType or str, optional): The logical type that should be assigned
            to the series. If no value is provided, the LogicalType for the series will
            be inferred.
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
        null_invalid_values (bool, optional): If True, replaces any invalid values with null. Defaults to False.

    Returns:
        Series: A series with Woodwork typing information initialized
    """
    if not _is_series(series):
        if (
            isinstance(series, (np.ndarray, pd.api.extensions.ExtensionArray))
            and series.ndim == 1
        ):
            series = pd.Series(series)
        elif isinstance(series, np.ndarray) and series.ndim != 1:
            raise ValueError(
                f"np.ndarray input must be 1 dimensional. Current np.ndarray is {series.ndim} dimensional",
            )
        else:
            raise TypeError(
                f"Input must be of series type. The current input is of type {type(series)}",
            )
    logical_type = _get_column_logical_type(series, logical_type, series.name)
    new_series = logical_type.transform(series, null_invalid_values=null_invalid_values)
    new_series.ww.init(
        logical_type=logical_type,
        semantic_tags=semantic_tags,
        use_standard_tags=use_standard_tags,
        description=description,
        origin=origin,
        metadata=metadata,
    )
    return new_series


def _is_series(data):
    if isinstance(data, pd.Series):
        return True
    elif _is_dask_series(data):
        return True
    elif _is_spark_series(data):
        return True
    return False


def _is_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return True
    elif _is_dask_dataframe(data):
        return True
    elif _is_spark_dataframe(data):
        return True
    return False


def get_invalid_schema_message(dataframe, schema):
    """Return a message indicating the reason that the provided schema cannot be used to
    initialize Woodwork on the dataframe. If the schema is valid for the dataframe,
    None will be returned.

    Args:
        dataframe (DataFrame): The dataframe against which to check the schema.
        schema (ww.TableSchema): The schema to use in the validity check.

    Returns:
        str or None: The reason that the schema is invalid for the dataframe
    """
    dataframe_cols = set(dataframe.columns)
    schema_cols = set(schema.columns.keys())

    df_cols_not_in_schema = dataframe_cols - schema_cols
    if df_cols_not_in_schema:
        return (
            f"The following columns in the DataFrame were missing from the typing information: "
            f"{df_cols_not_in_schema}"
        )
    schema_cols_not_in_df = schema_cols - dataframe_cols
    if schema_cols_not_in_df:
        return (
            f"The following columns in the typing information were missing from the DataFrame: "
            f"{schema_cols_not_in_df}"
        )
    logical_types = schema.logical_types
    for name in dataframe.columns:
        df_dtype = dataframe[name].dtype
        valid_dtype = logical_types[name]._get_valid_dtype(type(dataframe[name]))
        if str(df_dtype) != valid_dtype:
            return (
                f"dtype mismatch for column {name} between DataFrame dtype, "
                f"{df_dtype}, and {logical_types[name]} dtype, {valid_dtype}"
            )
    if schema.index is not None and isinstance(dataframe, pd.DataFrame):
        # Index validation not performed for Dask/Spark
        if not pd.Series(dataframe.index, dtype=dataframe[schema.index].dtype).equals(
            pd.Series(dataframe[schema.index].values),
        ):
            return "Index mismatch between DataFrame and typing information"
        elif not dataframe[schema.index].is_unique:
            return "Index column is not unique"
        elif dataframe[schema.index].isnull().any():
            return "Index contains null values"


def is_schema_valid(dataframe, schema):
    """Check if a schema is valid for initializing Woodwork on a dataframe

    Args:
        dataframe (DataFrame): The dataframe against which to check the schema.
        schema (ww.TableSchema): The schema to use in the validity check.

    Returns:
        boolean: Boolean indicating whether the schema is valid for the dataframe
    """

    invalid_schema_message = get_invalid_schema_message(dataframe, schema)
    if invalid_schema_message:
        return False
    return True


def _is_dask_series(data):
    if dd and isinstance(data, dd.Series):
        return True
    return False


def _is_dask_dataframe(data):
    if dd and isinstance(data, dd.DataFrame):
        return True
    return False


def _is_spark_dataframe(data):
    if ps and isinstance(data, ps.DataFrame):
        return True
    return False


def _is_spark_series(data):
    if ps and isinstance(data, ps.Series):
        return True
    return False


def _check_column_schema(method):
    """Decorator for WoodworkColumnAccessor that checks schema initialization"""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._schema is None:
            msg = (
                "Woodwork not initialized for this Series. Initialize by "
                "calling Series.ww.init"
            )
            raise WoodworkNotInitError(msg)
        return method(self, *args, **kwargs)

    return wrapper


def _check_table_schema(method):
    """Decorator for WoodworkTableAccessor that checks schema initialization"""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._schema is None:
            msg = (
                "Woodwork not initialized for this DataFrame. Initialize by "
                "calling DataFrame.ww.init"
            )
            raise WoodworkNotInitError(msg)
        diff_cols = set(self._dataframe.columns).difference(
            set(self._schema.columns.keys()),
        )
        if diff_cols:
            raise ColumnNotPresentInSchemaError(sorted(list(diff_cols)))
        return method(self, *args, **kwargs)

    return wrapper
