import warnings
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_mutual_info_score

from ._bin_numeric_cols_into_categories import _bin_numeric_cols_into_categories

from woodwork.accessor_utils import _is_dask_dataframe, _is_koalas_dataframe
from woodwork.exceptions import SparseDataWarning
from woodwork.utils import _update_progress, get_valid_mi_types


def _get_dependence_dict(
    dataframe,
    measure,
    num_bins=10,
    nrows=None,
    include_index=False,
    callback=None,
    extra_stats=False,
    min_shared=25,
    random_seed=0,
):
    """Calculates dependence measures between all pairs of columns in the DataFrame that
    support measuring dependence. Supports boolean, categorical, datetime, and numeric data.
    Call woodwork.utils.get_valid_dependence_types for a complete list of supported Logical Types.

    Args:
        dataframe (pd.DataFrame): Data containing Woodwork typing information
            from which to calculate dependence.
        measure (list or str): which dependence measures to calculate.
            A list of measures can be provided to calculate multiple
            measures at once.  Valid measure strings:

                - "pearson": calculates the Pearson correlation coefficient
                - "mutual": calculates the mutual information between columns
                - "max":  max(abs(pearson), mutual) for each pair of columns
                - "all": includes columns for "pearson", "mutual", and "max"
        num_bins (int): Determines number of bins to use for converting
            numeric features into categorical.
        nrows (int): The number of rows to sample for when determining dependence.
            If specified, samples the desired number of rows from the data.
            Defaults to using all rows.
        include_index (bool): If True, the column specified as the index will be
            included as long as its LogicalType is valid for measuring dependence.
            If False, the index column will not be considered. Defaults to False.
        callback (callable, optional): function to be called with incremental updates. Has the following parameters:

            - update (int): change in progress since last call
            - progress (int): the progress so far in the calculations
            - total (int): the total number of calculations to do
            - unit (str): unit of measurement for progress/total
            - time_elapsed (float): total time in seconds elapsed since start of call
        extra_stats (bool):  if True, additional column "shared_rows"
            recording the number of shared non-null rows for a column
            pair will be included with the dataframe.  If the "max"
            measure is being used, a "measure_used" column will be added
            that records whether Pearson or mutual information was the
            maximum dependence for a particular row.
        min_shared (int): the number of shared non-null rows needed to
            calculate.  Less rows than this will be considered too sparse
            to measure accurately and will return a NaN value. Must be
            non-negative.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    Returns:
        list(dict): A list containing dictionaries that have keys `column_1`,
        `column_2`, and keys for the specified dependence measures. The list is
        sorted in decending order by the first specified measure.
        Dependence information values are between 0 (no dependence) and 1
        (perfect dependency). For Pearson, values range from -1 to 1 but 0 is
        still no dependence.
    """
    start_time = timer()
    if not isinstance(measure, list):
        measure = [measure]
    if measure == ["all"]:
        measure = ["max", "mutual", "pearson"]

    if "max" in measure:
        calc_mutual = calc_pearson = calc_max = True
    else:
        calc_max = False
        calc_mutual = "mutual" in measure
        calc_pearson = "pearson" in measure

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
    # cut off data if necessary
    if _is_dask_dataframe(data):
        data = data.compute()
    elif _is_koalas_dataframe(dataframe):
        data = data.to_pandas()
    if nrows is not None and nrows < data.shape[0]:
        data = data.sample(nrows, random_state=random_seed)

    notna_mask = data.notnull()
    not_null_cols = data.columns[notna_mask.any()]
    if set(not_null_cols) != set(valid_columns):
        data = data.loc[:, not_null_cols]

    # Setup for progress callback and make initial call
    # Assume 1 unit for preprocessing, 2n for make categorical and (n*n+n)/2 for main calculation loop
    n = len(data.columns)
    total_loops = 1 + 2 * n + (n * n + n) / 2
    current_progress = _update_progress(
        start_time, timer(), 1, 0, total_loops, unit, callback
    )

    data = _bin_numeric_cols_into_categories(dataframe.ww.schema, data, num_bins)
    current_progress = _update_progress(
        start_time, timer(), 2 * n, current_progress, total_loops, unit, callback
    )

    # calculate mutual info for all pairs of columns
    dependence_list = []
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
                result = {"column_1": a_col, "column_2": b_col}
                num_intersect = (notna_mask[a_col] & notna_mask[b_col]).sum()
                too_sparse = num_intersect < min_shared
                if too_sparse:
                    # TODO: reword since Pearson can return NaN naturally
                    warnings.warn(
                        "One or more values in the returned matrix are NaN. A "
                        "NaN value indicates there were not enough rows where "
                        "both columns had non-null data",
                        SparseDataWarning,
                    )
                    result.update({measure_type: np.nan for measure_type in measure})
                else:
                    num_union = (notna_mask[a_col] | notna_mask[b_col]).sum()
                    if calc_mutual:
                        mi_score = adjusted_mutual_info_score(data[a_col], data[b_col])
                        mi_score = mi_score * num_intersect / num_union
                        if "mutual" in measure:
                            result["mutual"] = mi_score
                    if calc_pearson:
                        pearson_score = np.corrcoef(data[a_col], data[b_col])[0, 1]
                        pearson_score = pearson_score * num_intersect / num_union
                        if "pearson" in measure:
                            result["pearson"] = pearson_score
                    if calc_max:
                        score = pd.Series(
                            [mi_score, abs(pearson_score)], index=["mutual", "pearson"]
                        )
                        result["max"] = score.max()
                        if extra_stats:
                            result["measure_used"] = score.idxmax()

                if extra_stats:
                    result["shared_rows"] = num_intersect

                dependence_list.append(result)
                current_progress = _update_progress(
                    start_time,
                    timer(),
                    1,
                    current_progress,
                    total_loops,
                    unit,
                    callback,
                )

    def sort_key(result):
        key = result[measure[0]]
        if np.isnan(key):
            key = -2
        return key

    dependence_list.sort(key=sort_key, reverse=True)

    return dependence_list
