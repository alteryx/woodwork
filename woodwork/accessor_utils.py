import pandas as pd

from woodwork.logical_types import Datetime
from woodwork.type_sys.utils import _get_ltype_class
from woodwork.utils import _get_column_logical_type


def _update_dtype(series, logical_type):
    """Update the dtype of the series to match the dtype corresponding
    to the specified LogicalType. Raises an error if the conversion is not
    possible. Returns a series of the proper dtype."""

    if logical_type.pandas_dtype != str(series.dtype):
        # Try to update the series dtype - raise error if unsuccessful
        try:
            if _get_ltype_class(logical_type) == Datetime:
                # TODO: Uncomment when dask/koalas support is added for accessor
                # if dd and isinstance(self._series, dd.Series):
                #     name = self._series.name
                #     self._series = dd.to_datetime(self._series, format=self.logical_type.datetime_format)
                #     self._series.name = name
                # elif ks and isinstance(self._series, ks.Series):
                #     self._series = ks.Series(ks.to_datetime(self._series.to_numpy(),
                #                                             format=self.logical_type.datetime_format),
                #                             name=self._series.name)
                # else:
                series = pd.to_datetime(series, format=logical_type.datetime_format)
            else:
                # TODO: Uncomment when dask/koalas support is added for accessor
                # if ks and isinstance(self._series, ks.Series) and self.logical_type.backup_dtype:
                #     new_dtype = self.logical_type.backup_dtype
                # else:
                new_dtype = logical_type.pandas_dtype
                series = series.astype(new_dtype)
        except (TypeError, ValueError):
            error_msg = f'Error converting datatype for {series.name} from type {str(series.dtype)} ' \
                f'to type {logical_type.pandas_dtype}. Please confirm the underlying data is consistent with ' \
                f'logical type {logical_type}.'
            raise TypeError(error_msg)

    return series


def init_series(series, logical_type=None, semantic_tags=None,
                use_standard_tags=True, description=None, metadata=None):
    """Initializes Woodwork typing information for a Series, returning a new Series. The dtype
    of the series will be converted to match the dtype associated with the LogicalType.

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

    new_series = _update_dtype(series, logical_type)
    new_series.ww.init(logical_type=logical_type,
                       semantic_tags=semantic_tags,
                       use_standard_tags=use_standard_tags,
                       description=description,
                       metadata=metadata)
    return new_series
