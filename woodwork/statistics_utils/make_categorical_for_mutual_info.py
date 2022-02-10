import pandas as pd


def _make_categorical_for_mutual_info(schema, data, num_bins):
    """Transforms dataframe columns into numeric categories so that
    mutual information can be calculated

    Args:
        schema (woodwork.TableSchema): Woodwork typing info for the data
        data (pd.DataFrame): dataframe to use for calculating mutual information
        num_bins (int): Determines number of bins to use for converting
            numeric features into categorical.

    Returns:
        pd.DataFrame: data with values transformed and binned into numeric categorical values
    """

    for col_name in data.columns:
        column = schema.columns[col_name]
        if column.is_numeric:
            # bin numeric features to make categories
            data[col_name] = pd.qcut(data[col_name], num_bins, duplicates="drop")
        # Convert Datetimes to total seconds - an integer - and bin
        if column.is_datetime:
            data[col_name] = pd.qcut(
                data[col_name].view("int64"), num_bins, duplicates="drop"
            )
        # convert categories to integers
        new_col = data[col_name]
        if str(new_col.dtype) != "category":
            new_col = new_col.astype("category")
        data[col_name] = new_col.cat.codes
    return data
