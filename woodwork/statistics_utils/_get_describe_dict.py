import math
from timeit import default_timer as timer
from typing import Any, Callable, Dict, Sequence

import pandas as pd

from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.logical_types import (
    Age,
    AgeNullable,
    Datetime,
    Integer,
    IntegerNullable,
    LatLong,
    Unknown,
)
from woodwork.statistics_utils._get_histogram_values import _get_histogram_values
from woodwork.statistics_utils._get_mode import _get_mode
from woodwork.statistics_utils._get_numeric_value_counts_in_range import (
    _get_numeric_value_counts_in_range,
)
from woodwork.statistics_utils._get_recent_value_counts import _get_recent_value_counts
from woodwork.statistics_utils._get_top_values_categorical import (
    _get_top_values_categorical,
)
from woodwork.utils import CallbackCaller, _is_latlong_nan


def percentile(N, percent, count):
    """
    Source: https://stackoverflow.com/questions/2374640/how-do-i-calculate-percentiles-with-python-numpy/2753343#2753343

    Find the percentile of a list of values.

    Args:
        N (pd.Series): Series is a list of values. Note N MUST BE already sorted.
        percent (float): float value from 0.0 to 1.0.
        count (int): Count of values in series

    Returns:
        Value at percentile.
    """
    k = (count - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return N[int(k)]
    d0 = N.iat[int(f)] * (c - k)
    d1 = N.iat[int(c)] * (k - f)
    return d0 + d1


def _get_describe_dict(
    dataframe: pd.DataFrame,
    include: Sequence[str] = None,
    callback: Callable[[int, int, int, str, float], Any] = None,
    results_callback: Callable[[pd.DataFrame, pd.Series], Any] = None,
    extra_stats: bool = False,
    bins: int = 10,
    top_x: int = 10,
    recent_x: int = 10,
) -> Dict[str, dict]:
    """Calculates statistics for data contained in a DataFrame using Woodwork typing information.

    Args:
        dataframe (pd.DataFrame): DataFrame to be described with Woodwork typing information initialized
        include (list[str or LogicalType], optional): filter for what columns to include in the
            statistics returned. Can be a list of column names, semantic tags, logical types, or a list
            combining any of the three. It follows the most broad specification. Favors logical types
            then semantic tag then column name. If no matching columns are found, an empty DataFrame
            will be returned.
        callback (callable, optional): function to be called with incremental updates. Has the following parameters:

            - update (int): change in progress since last call
            - progress (int): the progress so far in the calculations
            - total (int): the total number of calculations to do
            - unit (str): unit of measurement for progress/total
            - time_elapsed (float): total time in seconds elapsed since start of call
        results_callback (callable, optional): function to be called with intermediate results. Has the following parameters:

            - results_so_far (pd.DataFrame): the full dataframe calculated so far
            - most_recent_calculation (pd.Series): the calculations for the most recent column

        extra_stats (bool): If True, will calculate a histogram for numeric columns, top values
            for categorical columns and value counts for the most recent values in datetime columns. Will also
            calculate value counts within the range of values present for integer columns if the range of
            values present is less than or equal to than the number of bins used to compute the histogram.
            Output can be controlled by bins, top_x and recent_x parameters.
        bins (int): Number of bins to use when calculating histogram for numeric columns. Defaults to 10.
            Will be ignored unless extra_stats=True.
        top_x (int): Number of items to return when getting the most frequently occurring values for categorical
            columns. Defaults to 10. Will be ignored unless extra_stats=True.
        recent_x (int): Number of values to return when calculating value counts for the most recent dates in
            datetime columns. Defaults to 10. Will be ignored unless extra_stats=True.

    Returns:
        dict[str -> dict]: A dictionary with a key for each column in the data or for each column
        matching the logical types, semantic tags or column names specified in ``include``, paired
        with a value containing a dictionary containing relevant statistics for that column.
    """
    start_time = timer()
    unit = "calculations"
    if include is not None:
        filtered_cols = dataframe.ww._filter_cols(include, col_names=True)
        cols_to_include = [
            (k, v) for k, v in dataframe.ww.columns.items() if k in filtered_cols
        ]
    else:
        cols_to_include = dataframe.ww.columns.items()

    results = {}

    if _is_dask_dataframe(dataframe):
        df = dataframe.compute()
    elif _is_spark_dataframe(dataframe):
        df = dataframe.to_pandas()

        # Any LatLong columns will be using lists, which we must convert
        # back to tuples so we can calculate the mode, which requires hashable values
        latlong_columns = [
            col_name
            for col_name, col in dataframe.ww.columns.items()
            if type(col.logical_type) == LatLong
        ]
        df[latlong_columns] = df[latlong_columns].applymap(
            lambda latlong: tuple(latlong) if latlong else latlong,
        )
    else:
        df = dataframe

    # Setup for progress callback and make initial call
    # Assume 1 unit for general preprocessing, plus main loop over column
    total_loops = 1 + len(cols_to_include)
    callback_caller = CallbackCaller(callback, unit, total_loops, start_time=start_time)
    callback_caller.update(1)

    for column_name, column in cols_to_include:
        if "index" in column.semantic_tags:
            callback_caller.update(1)
            continue
        values = {}
        logical_type = column.logical_type
        semantic_tags = column.semantic_tags
        series = df[column_name]

        agg_stats_to_calculate = {
            "category": ["count", "nunique"],
            Datetime: ["count", "max", "min", "nunique", "mean"],
            Unknown: ["count", "nunique"],
        }

        # Calculate Aggregation Stats
        if column.is_categorical:
            agg_stats = agg_stats_to_calculate["category"]
        elif column.is_numeric:
            agg_stats = None
        elif column.is_datetime:
            agg_stats = agg_stats_to_calculate[Datetime]
        elif column.is_unknown:
            agg_stats = agg_stats_to_calculate[Unknown]
        else:
            agg_stats = ["count"]
        values = {}
        if agg_stats:
            values = series.agg(agg_stats).to_dict()

        mode = _get_mode(series)
        # The format of the mode should match its format in the DataFrame
        if _is_spark_dataframe(dataframe) and series.name in latlong_columns:
            mode = list(mode)

        if column.is_latlong:
            nan_count = series.apply(_is_latlong_nan).sum()
            count = len(series) - nan_count

            values["nan_count"] = nan_count
            values["count"] = count
        else:
            values["nan_count"] = series.isna().sum()

        # Calculate other specific stats based on logical type or semantic tags
        if column.is_boolean:
            values["num_false"] = series.value_counts().get(False, 0)
            values["num_true"] = series.value_counts().get(True, 0)
        elif column.is_numeric:
            if values["nan_count"] == 0:
                agg_stats = ["count", "nunique", "mean", "std"]
                values.update(series.agg(agg_stats).to_dict())
                series = series.sort_values(
                    ignore_index=True,
                )
                values["max"] = series.iat[int(values["count"] - 1)]
                values["min"] = series.iat[0]
                values["first_quartile"] = percentile(
                    series,
                    0.25,
                    int(values["count"]),
                )
                values["second_quartile"] = percentile(
                    series,
                    0.5,
                    int(values["count"]),
                )
                values["third_quartile"] = percentile(
                    series,
                    0.75,
                    int(values["count"]),
                )
            else:
                agg_stats = ["count", "max", "min", "nunique", "mean", "std"]
                values.update(series.agg(agg_stats).to_dict())
                quant_values = series.quantile([0.25, 0.5, 0.75]).tolist()
                values["first_quartile"] = quant_values[0]
                values["second_quartile"] = quant_values[1]
                values["third_quartile"] = quant_values[2]

        values["mode"] = mode
        values["physical_type"] = series.dtype
        values["logical_type"] = logical_type
        values["semantic_tags"] = semantic_tags

        # Calculate extra detailed stats, if requested
        if extra_stats:
            if column.is_numeric:
                if pd.isnull(values["max"]) or pd.isnull(values["min"]):
                    values["histogram"] = []
                    values["top_values"] = []
                else:
                    values["histogram"] = _get_histogram_values(series, bins=bins)
                    if isinstance(
                        column.logical_type,
                        (Age, AgeNullable, Integer, IntegerNullable),
                    ):
                        _range = range(int(values["min"]), int(values["max"]) + 1)
                        # Calculate top numeric values if range of values present
                        # is less than or equal number of histogram bins
                        range_len = int(values["max"]) + 1 - int(values["min"])
                        if range_len <= bins and (series % 1 == 0).all():
                            values["top_values"] = _get_numeric_value_counts_in_range(
                                series,
                                _range,
                            )
            elif column.is_categorical:
                values["top_values"] = _get_top_values_categorical(series, top_x)
            elif column.is_datetime:
                values["recent_values"] = _get_recent_value_counts(series, recent_x)

        results[column_name] = values
        if results_callback is not None:
            results_so_far = pd.DataFrame.from_dict(results)
            most_recent_calculations = pd.Series(values, name=column_name)
            results_callback(results_so_far, most_recent_calculations)
        callback_caller.update(1)
    return results
