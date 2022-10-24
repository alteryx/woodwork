from collections import namedtuple

import numpy as np
import pandas as pd
from statsmodels.stats.stattools import medcouple

method_result = namedtuple(
    "MedcoupleHeuristicResult",
    (
        "method",
        "mc",
    ),
)


def _determine_best_outlier_method(series):
    """Determines the best outlier method to use based on the distribution of the series and outcome of the medcouple statistic.

    Args:
        series (Series): Data on which the medcouple statistic will be run in order to determine skewness.

    Note:
        The calculation of the medcouple statistic has a large memory requirement of O(N**2), therefore larger series will
        have a random subset selected in order to determine skewness and the best outlier method.

    Returns:
        MedcoupleHeuristicResult - Named tuple with 2 fields
            method (str): Name of the outlier method to use.
            mc (float): The medcouple statistic (if the method chosen is medcouple, otherwise None).
    """
    np.random.seed(42)
    series_size = len(series)
    if series_size < 10000:
        series_sample = series.copy()
    else:
        series_sample = np.random.choice(series, 10000, replace=False)
    method = "medcouple"
    col = pd.Series(series_sample)
    mc = medcouple(col)
    if np.abs(mc) < 0.2:
        method = "box_plot"
        mc = None
    return method_result(method, mc)
