import pandas as pd


def _bin_numeric_cols_into_categories(schema, data, num_bins):
    """Transforms dataframe columns into numeric categories so that
    dependence can be calculated.

    Args:
        schema (woodwork.TableSchema): Woodwork typing info for the data.
        data (dict[pd.Series]): Dictionary of Pandas series to use for
            calculating dependence.
        num_bins (int): Determines number of bins to use for converting
            numeric features into categorical.

    Returns:
        None
    """
    for col_name in data.keys():
        column = schema.columns[col_name]
        if (column.is_numeric or column.is_datetime) and len(set(data[col_name])) > 2:
            # bin numeric features to make categories when there are enough values
            data[col_name] = pd.qcut(data[col_name], num_bins, duplicates="drop")
        # convert categories to integers
        new_col = data[col_name]
        if str(new_col.dtype) != "category":
            new_col = new_col.astype("category")
        data[col_name] = new_col.cat.codes
