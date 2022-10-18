import itertools
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.cluster import adjusted_mutual_info_score

from woodwork.exceptions import SparseDataWarning


def _calculate_dependence_measure(
    measure,
    data,
    results,
    callback_caller,
    notna_mask,
    min_shared,
    col_names,
    target_col,
):
    """
    Calculates the specified dependence measure for each pair of columns in the
    provided list of columns.  If two columns do not have enough shared rows
    (determined using the notna_mask) no measurement is calculated for that pair.
    Updates the results dictionary.

    Args:
        measure (str): Dependence measure to calculate.
        data (dict[pd.Series]): Dictionary of pandas series to measure.
        results (defaultdict[dict]): dictionary to store results
        callback_caller (CallbackCaller): Callback calling object.
        notna_mask (pd.DataFrame): Boolean mask of original data, shows whether
            a cell is null or not.
        min_shared (int): Mininum rows of shared data to calculate dependence.
        col_names (list): List of columns to use for this calculation.
        target_col (str): The string name of the target column.

    Returns:
        None
    """
    column_pairs = []
    if target_col is None:
        column_pairs = itertools.combinations(col_names, 2)
    elif target_col in col_names:
        column_pairs = [(x, target_col) for x in col_names if x != target_col]
    for a_col, b_col in column_pairs:
        result = results[(a_col, b_col)]
        # check if result already has keys, meaning function has been called
        # previously for a different measure and some computation can be skipped
        if "column_1" in result:
            num_intersect = result["shared_rows"]
        else:
            result["column_1"] = a_col
            result["column_2"] = b_col
            num_intersect = (notna_mask[a_col] & notna_mask[b_col]).sum()
            result["shared_rows"] = num_intersect

        too_sparse = num_intersect < min_shared
        if too_sparse:
            warnings.warn(
                "One or more pairs of columns did not share enough rows of "
                "non-null data to measure the relationship.  The measurement"
                " for these columns will be NaN.  Use 'extra_stats=True' to "
                "get the shared rows for each pair of columns.",
                SparseDataWarning,
            )
            result[measure] = np.nan
        else:
            if "num_union" in result:
                num_union = result["num_union"]
            else:
                num_union = (notna_mask[a_col] | notna_mask[b_col]).sum()
                result["num_union"] = num_union
            intersect = notna_mask[a_col] & notna_mask[b_col]
            if measure == "mutual_info":
                score = adjusted_mutual_info_score(
                    data[a_col][intersect],
                    data[b_col][intersect],
                )
            elif measure == "pearson":
                score = np.corrcoef(data[a_col][intersect], data[b_col][intersect])[
                    0,
                    1,
                ]
            elif measure == "spearman":
                score, _ = spearmanr(data[a_col][intersect], data[b_col][intersect])

            score = score * num_intersect / num_union
            result[measure] = score
        # increment progress in either case
        callback_caller.update(1)
