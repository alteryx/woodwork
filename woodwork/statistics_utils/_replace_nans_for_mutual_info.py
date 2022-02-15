from ._get_mode import _get_mode

from woodwork.logical_types import Double


def _replace_nans_for_mutual_info(schema, data):
    """Replace NaN values in the dataframe so that mutual information can be calculated

    Args:
        schema (woodwork.TableSchema): Woodwork typing info for the data
        data (pd.DataFrame): dataframe to use for calculating mutual information

    Returns:
        pd.DataFrame: data with nans replaced with either mean or mode

    """
    for column_name in data.columns[data.isnull().any()]:
        column = schema.columns[column_name]
        series = data[column_name]

        if column.is_numeric or column.is_datetime:
            mean = series.mean()
            if (
                isinstance(mean, float)
                and not mean.is_integer()
                and not type(column.logical_type) == Double
            ):
                data[column_name] = series.astype("float")
            data[column_name] = data[column_name].fillna(mean)
        elif column.is_categorical or column.is_boolean:
            mode = _get_mode(series)
            data[column_name] = series.fillna(mode)
    return data
