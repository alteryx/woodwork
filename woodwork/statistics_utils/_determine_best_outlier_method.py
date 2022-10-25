from collections import namedtuple

import numpy as np
import pandas as pd

import woodwork as ww
from woodwork.statistics_utils._get_medcouple_statistic import _get_medcouple


def _sample_for_medcouple(series, seed=42):
    """Samples the data in order to calculate the medcouple statistic.

    Args:
        series (Series): Data on which sampling will occur.
        seed (int): Seed for random sampling.

    Note:
        The calculation of the medcouple statistic has a large memory requirement of O(N**2), therefore a series over 10,000
        observations will be randomly sampled up to 10,000.

    Returns:
        float: Medcouple statistic from the sampled data
    """
    np.random.seed(seed)
    series_size = len(series)
    if series_size < ww.config.get_option("medcouple_sample_size"):
        series_sample = series.copy()
    else:
        series_sample = np.random.choice(
            series, ww.config.get_option("medcouple_sample_size"), replace=False
        )
    col = pd.Series(series_sample)
    mc = _get_medcouple(col)
    return mc


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
    mc = _sample_for_medcouple(series)
    method = "medcouple"
    if np.abs(mc) < ww.config.get_option("medcouple_threshold"):
        method = "box_plot"
        mc = None
    return method_result(method, mc)
