import pandas as pd

from woodwork.exceptions import TypeConversionError
from woodwork.logical_types import Datetime, LatLong, Ordinal
from woodwork.type_sys.utils import _get_ltype_class
from woodwork.utils import (
    _get_column_logical_type,
    _reformat_to_latlong,
    import_or_none
)

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def init_series(series, logical_type=None, semantic_tags=None,
                use_standard_tags=True, description=None, metadata=None):
    """Initializes Woodwork typing information for a Series, returning a new Series. The dtype
    of the returned series will be converted to match the dtype associated with the LogicalType.

    Args:
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
        metadata (dict[str -> json serializable], optional): Metadata associated with the series.

    Returns:
        Series: A series with Woodwork typing information initialized
    """
    logical_type = _get_column_logical_type(series, logical_type, series.name)

    new_series = _update_column_dtype(series, logical_type)
    new_series.ww.init(logical_type=logical_type,
                       semantic_tags=semantic_tags,
                       use_standard_tags=use_standard_tags,
                       description=description,
                       metadata=metadata)
    return new_series


def _update_column_dtype(series, logical_type):
    """Update the dtype of the underlying series to match the dtype corresponding
    to the LogicalType for the column."""
    if isinstance(logical_type, Ordinal):
        logical_type._validate_data(series)
    if _get_ltype_class(logical_type) == LatLong:
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
            error_msg = f'Error converting datatype for {series.name} from type {str(series.dtype)} ' \
                f'to type {logical_type.pandas_dtype}. Please confirm the underlying data is consistent with ' \
                f'logical type {logical_type}.'
            raise TypeConversionError(error_msg)
    return series
