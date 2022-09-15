def _convert_ordinal_to_numeric(schema, data):
    """Transforms ordinal data to numeric data so that dependence can be calculated.

    Args:
        schema (woodwork.TableSchema): Woodwork typing info for the data.
        data (dict[pd.Series]): Dictionary of Pandas series to use for
            calculating dependence.

    Returns:
        None
    """
    for col_name in data.keys():
        column = schema.columns[col_name]
        if column.is_ordinal:
            order = list(column.logical_type.order)
            data[col_name] = data[col_name].apply(lambda x: order.index(x))
