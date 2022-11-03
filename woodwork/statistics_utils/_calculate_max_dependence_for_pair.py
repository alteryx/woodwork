import numpy as np
import pandas as pd


def _calculate_max_dependence_for_pair(result, min_shared, extra_stats):
    """Calculates the max dependence measure for a pair of columns. While
    the magnitude of a negatively correlated value is used to determine the max,
    the original (negative) value will be used in the "max" column.

    Args:
        result (dict): Dictionary storing dependence measurements, etc.
            for a pair of columns
        min_shared (int): Mininum rows of shared data to calculate dependence.
        extra_stats (bool):  If True, additional records will be included in the
            finalized result dictionary:

                - shared_rows (int): The number of shared non-null rows for a column pair.
                - measure_used (str): Which measure was the max for a column pair.

    Returns:
        None
    """
    # if pearson was not measured, mutual info must be max, since columns valid
    # for pearson are a subset of columns valid for mutual info
    if "pearson" in result or "spearman" in result:
        score = pd.Series(
            [
                result["mutual_info"],
                abs(result.get("pearson", np.nan)),
                abs(result.get("spearman", np.nan)),
            ],
            index=["mutual_info", "pearson", "spearman"],
        )
        # to keep sign, get name of max score and use original value
        measure_used = score.idxmax()
    else:
        measure_used = "mutual_info"

    # if all measures were nan, measure_used will be nan (float)
    result["max"] = result.get(measure_used, np.nan)

    if extra_stats:
        if result["shared_rows"] < min_shared:
            result["measure_used"] = "too sparse"
        else:
            result["measure_used"] = measure_used
