from woodwork.utils import get_valid_mi_types


def _get_valid_mi_columns(dataframe, include_index=False, include_time_index=False):
    """Returns a list of columns from the DataFrame with valid
        Logical Types that support mutual information.

    Args:
        dataframe (pd.DataFrame): Data containing Woodwork typing information
            from which to calculate mutual information.
        include_index (bool): If True, the column specified as the index will be
            included as long as its LogicalType is valid for mutual information calculations.
            If False, the index column will not have mutual information calculated for it.
            Defaults to False.
        include_index (bool): If True, the column specified as the time index will be
            included for mutual information calculations.
            If False, the time index column will not have mutual information calculated for it.
            Defaults to False.

    Returns:
        list: A list of column names that have valid Logical Types that support
        mutual information.
    """
    valid_types = tuple(get_valid_mi_types())
    valid_columns = [
        col_name
        for col_name, col in dataframe.ww.columns.items()
        if isinstance(col.logical_type, valid_types)
    ]
    if not include_index and dataframe.ww.index is not None:
        valid_columns.remove(dataframe.ww.index)
    if not include_time_index and dataframe.ww.time_index is not None:
        valid_columns.remove(dataframe.ww.time_index)

    return valid_columns
