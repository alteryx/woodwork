from woodwork.utils import import_or_none

ks = import_or_none('databricks.koalas')


def convert_series(series, logical_type):
    '''Converts a series to match the pandas_dtype of the provided logical type
    for pandas/Dask, or converts to the backup_dtype for Koalas if one is defined'''
    if ks and isinstance(series, ks.Series) and logical_type.backup_dtype:
        return series.astype(logical_type.backup_dtype)
    else:
        return series.astype(logical_type.pandas_dtype)
