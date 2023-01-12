import numpy as np
import pandas as pd

import woodwork as ww
from woodwork.statistics_utils._medcouple_implementation import (
    _calculate_medcouple_statistic,
)


def _get_medcouple_statistic(array_, axis=0):
    array_ = np.asarray(array_, dtype=np.double)
    mc = np.apply_along_axis(_calculate_medcouple_statistic, axis, array_)
    if isinstance(mc, np.ndarray) and isinstance(mc.tolist(), float):
        mc = mc.tolist()
        mc = round(mc, 3)
    return mc


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
            series,
            ww.config.get_option("medcouple_sample_size"),
            replace=False,
        )
    col = pd.Series(series_sample)
    mc = _get_medcouple_statistic(col)
    return mc
