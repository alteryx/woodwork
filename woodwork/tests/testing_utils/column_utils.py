from woodwork.accessor_utils import _get_dtype_to_convert
from woodwork.utils import import_or_none

ks = import_or_none('databricks.koalas')


def convert_series(series, logical_type):
    """Converts a series to match the pandas_dtype of the provided logical type
    for pandas/Dask, or converts to the backup_dtype for Koalas if one is defined"""
    convert_dtype = _get_dtype_to_convert(series, logical_type)
    return series.astype(convert_dtype)
