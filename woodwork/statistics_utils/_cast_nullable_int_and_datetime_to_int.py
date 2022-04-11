from woodwork.logical_types import IntegerNullable


def _cast_nullable_int_and_datetime_to_int(data_dict, woodwork_columns):
    for col_name in data_dict:
        column = woodwork_columns[col_name]
        if isinstance(column.logical_type, IntegerNullable):
            cur_dtype = data_dict[col_name].dtype
            new_dtype = cur_dtype.name.lower()  # e.g. Int64 -> int64
            data_dict[col_name] = data_dict[col_name].astype(new_dtype)
        if column.is_datetime:
            data_dict[col_name] = data_dict[col_name].view("int64")
