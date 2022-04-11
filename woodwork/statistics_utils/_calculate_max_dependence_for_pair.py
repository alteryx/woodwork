import numpy as np
import pandas as pd


def _calculate_max_dependence_for_pair(
    result, returned_measures, min_shared, extra_stats
):
    """Calculates the max dependence measure for a pair of columns. While
    the magnitude of a negatively correlated value is used to determine the max,
    the original (negative) value will be used in the "max" column.
    """
    # if pearson was not measured, mutual info must be max
    if "pearson" in result:
        score = pd.Series(
            [result["mutual_info"], abs(result["pearson"])],
            index=["mutual_info", "pearson"],
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
