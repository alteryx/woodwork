from timeit import default_timer as timer

from sklearn.metrics.cluster import normalized_mutual_info_score

from ._make_categorical_for_mutual_info import _make_categorical_for_mutual_info
from ._replace_nans_for_mutual_info import _replace_nans_for_mutual_info

from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.utils import _update_progress, get_valid_mi_types


def _get_mutual_information_dict(
    dataframe, num_bins=10, nrows=None, include_index=False, callback=None
):
    """Calculates mutual information between all pairs of columns in the DataFrame that
    support mutual information. Logical Types that support mutual information are as
    follows:  Boolean, Categorical, CountryCode, Datetime, Double, Integer, Ordinal,
    PostalCode, and SubRegionCode

    Args:
        dataframe (pd.DataFrame): Data containing Woodwork typing information
            from which to calculate mutual information.
        num_bins (int): Determines number of bins to use for converting
            numeric features into categorical.
        nrows (int): The number of rows to sample for when determining mutual info.
            If specified, samples the desired number of rows from the data.
            Defaults to using all rows.
        include_index (bool): If True, the column specified as the index will be
            included as long as its LogicalType is valid for mutual information calculations.
            If False, the index column will not have mutual information calculated for it.
            Defaults to False.
        callback (callable, optional): function to be called with incremental updates. Has the following parameters:

            - update (int): change in progress since last call
            - progress (int): the progress so far in the calculations
            - total (int): the total number of calculations to do
            - unit (str): unit of measurement for progress/total
            - time_elapsed (float): total time in seconds elapsed since start of call

    Returns:
        list(dict): A list containing dictionaries that have keys `column_1`,
        `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
        Mutual information values are between 0 (no mutual information) and 1
        (perfect dependency).
    """
    start_time = timer()
    unit = "calculations"
    valid_types = get_valid_mi_types()
    valid_columns = [
        col_name
        for col_name, col in dataframe.ww.columns.items()
        if type(col.logical_type) in valid_types
    ]

    index = dataframe.ww.index
    if not include_index and index is not None and index in valid_columns:
        valid_columns.remove(index)

    data = dataframe.loc[:, valid_columns]
    if _is_dask_dataframe(data):
        data = data.compute()
    if _is_spark_dataframe(dataframe):
        data = data.to_pandas()

    # cut off data if necessary
    if nrows is not None and nrows < data.shape[0]:
        data = data.sample(nrows)

    # remove fully null columns
    not_null_cols = data.columns[data.notnull().any()]
    if set(not_null_cols) != set(valid_columns):
        data = data.loc[:, not_null_cols]

    # Setup for progress callback and make initial call
    # Assume 1 unit for preprocessing, n for replace nans, n for make categorical and (n*n+n)/2 for main calculation loop
    n = len(data.columns)
    total_loops = 1 + 2 * n + (n * n + n) / 2
    current_progress = _update_progress(
        start_time, timer(), 1, 0, total_loops, unit, callback
    )

    data = _replace_nans_for_mutual_info(dataframe.ww.schema, data)
    current_progress = _update_progress(
        start_time, timer(), n, current_progress, total_loops, unit, callback
    )

    data = _make_categorical_for_mutual_info(dataframe.ww.schema, data, num_bins)
    current_progress = _update_progress(
        start_time, timer(), n, current_progress, total_loops, unit, callback
    )

    # calculate mutual info for all pairs of columns
    mutual_info = []
    col_names = data.columns.to_list()
    for i, a_col in enumerate(col_names):
        for j in range(i, len(col_names)):
            b_col = col_names[j]
            if a_col == b_col:
                # Ignore because the mutual info for a column with itself will always be 1
                current_progress = _update_progress(
                    start_time,
                    timer(),
                    1,
                    current_progress,
                    total_loops,
                    unit,
                    callback,
                )
                continue
            else:
                mi_score = normalized_mutual_info_score(data[a_col], data[b_col])
                mutual_info.append(
                    {"column_1": a_col, "column_2": b_col, "mutual_info": mi_score}
                )
                current_progress = _update_progress(
                    start_time,
                    timer(),
                    1,
                    current_progress,
                    total_loops,
                    unit,
                    callback,
                )
    mutual_info.sort(key=lambda mi: mi["mutual_info"], reverse=True)

    return mutual_info
