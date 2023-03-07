from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import skew

import woodwork as ww
from woodwork.statistics_utils._get_medcouple_statistic import _sample_for_medcouple
from woodwork.utils import import_or_none

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


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


def _determine_coefficients(series, mc):
    """Determines the coefficients to use in the medcouple bounds equation using the moment of skewness.

    Args:
        series (Series): Data on which the medcouple statistic will be run in order to determine coefficients.

    Note:
        Based off of the paper I. H. Adil and A. R. Irshad, "A Modified Approach for Detection of Outliers",
        Pak.j.stat.oper.res. Vol.XI No.1 2015 pp. 91-102, 2015.

    Returns:
        MedcoupleHeuristicResult - Named tuple with 2 fields
            method (str): Name of the outlier method to use.
            mc (float): The medcouple statistic (if the method chosen is medcouple, otherwise None).
    """
    try:
        coeff = np.abs(skew(series))
    except ValueError:
        # skew can't handle Int64 dtype
        coeff = np.abs(skew(series.astype("float64")))
    coeff = min(coeff, 3.5)
    if mc >= 0:
        return -coeff, coeff
    else:
        return coeff, -coeff


def _get_low_high_bound(series, method, q1, q3, min_value, max_value, mc=None):
    iqr = q3 - q1
    if method == "medcouple":
        if mc is None:
            raise ValueError(
                "If the method selected is medcouple, then mc cannot be None.",
            )
        # Medcouple coefficients change based on the skew direction
        lower_bound_coef, higher_bound_coef = _determine_coefficients(series, mc)
        low_bound = q1 - 1.5 * np.exp(lower_bound_coef * mc) * iqr
        high_bound = q3 + 1.5 * np.exp(higher_bound_coef * mc) * iqr
    elif method == "box_plot":
        low_bound = q1 - (iqr * 1.5)
        high_bound = q3 + (iqr * 1.5)
    else:
        raise ValueError(
            f"Acceptable methods are 'box_plot' and 'medcouple'. The value passed was '{method}'.",
        )
    # Box plot bounds calculation - the bounds should never be beyond the min and max values
    low_bound = max(low_bound, min_value)
    high_bound = min(high_bound, max_value)
    return low_bound, high_bound


def _get_box_plot_info_for_column(
    series,
    method="best",
    quantiles=None,
    include_indices_and_values=True,
    ignore_zeros=False,
):
    """Gets the information necessary to create a box and whisker plot with outliers for a numeric column.

    Args:
        series (Series): Data for which the box plot and outlier information will be gathered.
            Will be used to calculate quantiles if none are provided.
        method (str): The method to use when calculating the box and whiskers plot. Options are 'box_plot' and 'medcouple'.
        Defaults to 'best' at which point a heuristic will determine the appropriate method to use.
        quantiles (dict[float -> float], optional): A dictionary containing the quantiles for the data
            where the key indicates the quantile, and the value is the quantile's value for the data.
        include_indices_and_values (bool, optional): Whether or not the lists containing individual
            outlier values and their indices will be included in the returned dictionary.
            Defaults to True.
        ignore_zeros (bool): Whether to ignore 0 values (not NaN values) when calculating the box plot and outliers.
                Defaults to False.

    Note:
        The minimum quantiles necessary for building a box plot using the IQR method are the
        minimum value (0.0 in the quantiles dict), first quartile (0.25), third quartile (0.75), and maximum value (1.0).
        If no quantiles are provided, the following quantiles will be calculated:
        {0.0, 0.25, 0.5, 0.75, 1.0}, which correspond to {min, first quantile, median, third quantile, max}.

    Returns:
        (dict[str -> float,list[number]]): Returns a dictionary containing box plot information for the Series.
            The following elements will be found in the dictionary:

            - low_bound (float): the lowest data point in the dataset excluding any outliers - to be used as a whisker
            - high_bound (float): the highest point in the dataset excluding any outliers - to be used as a whisker
            - quantiles (list[float]): the quantiles used to determine the bounds.
                If quantiles were passed in, will contain all quantiles passed in. Otherwise, contains the five
                quantiles {0.0, 0.25, 0.5, 0.75, 1.0}.
            - low_values (list[float, int], optional): the values of the lower outliers.
                Will not be included if ``include_indices_and_values`` is False.
            - high_values (list[float, int], optional): the values of the upper outliers
                Will not be included if ``include_indices_and_values`` is False.
            - low_indices (list[int], optional): the corresponding index values for each of the lower outliers
                Will not be included if ``include_indices_and_values`` is False.
            - high_indices (list[int], optional): the corresponding index values for each of the upper outliers
                Will not be included if ``include_indices_and_values`` is False.
    """
    if not series.ww._schema.is_numeric:
        raise TypeError("Cannot calculate box plot statistics for non-numeric column")

    if quantiles and not isinstance(quantiles, dict):
        raise TypeError("quantiles must be a dictionary.")

    if dd and isinstance(series, dd.Series):
        series = series.compute()
    if ps and isinstance(series, ps.Series):
        series = series.to_pandas()

    # remove null values from the data
    series = series.dropna()

    if ignore_zeros:
        series = series[series.astype(bool)]

    outliers_dict = {}
    # An empty or fully null Series has no outliers, bounds, or quantiles
    if series.shape[0] == 0:
        if include_indices_and_values:
            outliers_dict = {
                "low_values": [],
                "high_values": [],
                "low_indices": [],
                "high_indices": [],
            }
        return {
            "low_bound": np.nan,
            "high_bound": np.nan,
            "quantiles": {
                0.0: np.nan,
                0.25: np.nan,
                0.5: np.nan,
                0.75: np.nan,
                1.0: np.nan,
            },
            **outliers_dict,
        }

    # calculate the outlier bounds using IQR
    if quantiles is None:
        quantiles = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
    elif len(set(quantiles.keys()) & {0.0, 0.25, 0.75, 1.0}) != 4:
        raise ValueError(
            "Input quantiles do not contain the minimum necessary quantiles for box plot calculation: "
            "0.0 (the minimum value), 0.25 (the first quartile), 0.75 (the third quartile), and 1.0 (the maximum value).",
        )
    min_value = quantiles[0.0]
    q1 = quantiles[0.25]
    q3 = quantiles[0.75]
    max_value = quantiles[1.0]

    mc = None
    if method == "best":
        method, mc = _determine_best_outlier_method(series)
    if method == "medcouple" and mc is None:
        mc = _sample_for_medcouple(series)

    if isinstance(mc, np.ndarray):
        mc = mc.tolist()

    low_bound, high_bound = _get_low_high_bound(
        series,
        method,
        q1,
        q3,
        min_value,
        max_value,
        mc,
    )
    low_bound = round(low_bound, 5) if isinstance(low_bound, float) else low_bound
    high_bound = round(high_bound, 5) if isinstance(high_bound, float) else high_bound

    if include_indices_and_values:
        # identify outliers in the series
        low_series = (
            series[series < low_bound]
            if low_bound > min_value
            else pd.Series(dtype="float64")
        )
        high_series = (
            series[series > high_bound]
            if high_bound < max_value
            else pd.Series(dtype="float64")
        )

        outliers_dict = {
            "low_values": low_series.tolist(),
            "high_values": high_series.tolist(),
            "low_indices": low_series.index.tolist(),
            "high_indices": high_series.index.tolist(),
        }

    outliers_dict["method"] = method
    if method == "medcouple":
        outliers_dict["medcouple_stat"] = mc

    return {
        "low_bound": low_bound,
        "high_bound": high_bound,
        "quantiles": quantiles,
        **outliers_dict,
    }
