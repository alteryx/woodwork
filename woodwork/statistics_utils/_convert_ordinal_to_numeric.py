def _convert_ordinal_to_numeric(schema, data):
    """Transforms ordinal data to numeric data so that dependence can be calculated.

    Args:
        schema (woodwork.TableSchema): Woodwork typing info for the data.
        data (dict[pd.Series]): Dictionary of Pandas series to use for
            calculating dependence.

    Returns:
        None
    """
    columns = [k for k, v in schema.logical_types.items() if "Ordinal" in str(v)]
    for col_name in columns:
        if col_name in data:
            order = list(schema.columns[col_name].logical_type.order)
            data[col_name] = data[col_name].apply(lambda x: order.index(x))
