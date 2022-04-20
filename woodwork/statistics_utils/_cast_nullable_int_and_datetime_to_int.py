from pandas.api.types import is_integer_dtype

from woodwork.logical_types import _NULLABLE_PHYSICAL_TYPES


def _cast_nullable_int_and_datetime_to_int(data_dict, woodwork_columns):
    for col_name in data_dict:
        column = woodwork_columns[col_name]
        series = data_dict[col_name]
        if (
            column.is_numeric
            and series.dtype.name in _NULLABLE_PHYSICAL_TYPES
            and is_integer_dtype(series.dtype)
        ):
            new_dtype = series.dtype.name.lower()  # e.g. Int64 -> int64
            data_dict[col_name] = series.astype(new_dtype)
        if column.is_datetime:
            data_dict[col_name] = series.view("int64")
