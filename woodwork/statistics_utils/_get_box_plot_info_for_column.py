import numpy as np
import pandas as pd

from woodwork.utils import import_or_none

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


def _get_box_plot_info_for_column(
    series, quantiles=None, include_indices_and_values=True
):
    """Gets the information necessary to create a box and whisker plot with outliers for a numeric column
        using the IQR method.

    Args:
        series (Series): Data for which the box plot and outlier information will be gathered.
            Will be used to calculate quantiles if none are provided.
        quantiles (dict[float -> float], optional): A dictionary containing the quantiles for the data
            where the key indicates the quantile, and the value is the quantile's value for the data.
        include_indices_and_values (bool, optional): Whether or not the lists containing individual
            outlier values and their indices will be included in the returned dictionary.
            Defaults to True.

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
            "0.0 (the minimum value), 0.25 (the first quartile), 0.75 (the third quartile), and 1.0 (the maximum value)."
        )
    min_value = quantiles[0.0]
    q1 = quantiles[0.25]
    q3 = quantiles[0.75]
    max_value = quantiles[1.0]

    # Box plot bounds calculation - the bounds should never be beyond the min and max values
    iqr = q3 - q1
    low_bound = max(q1 - (iqr * 1.5), min_value)
    high_bound = min(q3 + (iqr * 1.5), max_value)

    if include_indices_and_values:
        # identify outliers in the series
        low_series = (
            series[series < low_bound] if low_bound > min_value else pd.Series()
        )
        high_series = (
            series[series > high_bound] if high_bound < max_value else pd.Series()
        )

        outliers_dict = {
            "low_values": low_series.tolist(),
            "high_values": high_series.tolist(),
            "low_indices": low_series.index.tolist(),
            "high_indices": high_series.index.tolist(),
        }

    return {
        "low_bound": low_bound,
        "high_bound": high_bound,
        "quantiles": quantiles,
        **outliers_dict,
    }
