from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

from woodwork.accessor_utils import _is_dask_dataframe, _is_koalas_dataframe
from woodwork.logical_types import (
    Datetime,
    Double,
    Integer,
    IntegerNullable,
    LatLong
)
from woodwork.utils import _update_progress, get_valid_mi_types, import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def _get_describe_dict(dataframe, include=None, callback=None,
                       extra_stats=False, bins=10, top_x=10, recent_x=10):
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
    unit = 'calculations'
    agg_stats_to_calculate = {
        'category': ["count", "nunique"],
        'numeric': ["count", "max", "min", "nunique", "mean", "std"],
        Datetime: ["count", "max", "min", "nunique", "mean"],
    }
    if include is not None:
        filtered_cols = dataframe.ww._filter_cols(include, col_names=True)
        cols_to_include = [(k, v) for k, v in dataframe.ww.columns.items() if k in filtered_cols]
    else:
        cols_to_include = dataframe.ww.columns.items()

    results = {}

    if _is_dask_dataframe(dataframe):
        df = dataframe.compute()
    elif _is_koalas_dataframe(dataframe):
        df = dataframe.to_pandas()

        # Any LatLong columns will be using lists, which we must convert
        # back to tuples so we can calculate the mode, which requires hashable values
        latlong_columns = [col_name for col_name, col in dataframe.ww.columns.items() if type(col.logical_type) == LatLong]
        df[latlong_columns] = df[latlong_columns].applymap(lambda latlong: tuple(latlong) if latlong else latlong)
    else:
        df = dataframe

    # Setup for progress callback and make initial call
    # Assume 1 unit for general preprocessing, plus main loop over column
    total_loops = 1 + len(cols_to_include)
    current_progress = _update_progress(start_time, timer(), 1, 0, total_loops, unit, callback)

    for column_name, column in cols_to_include:
        if 'index' in column.semantic_tags:
            current_progress = _update_progress(start_time, timer(), 1, current_progress, total_loops, unit, callback)
            continue
        values = {}
        logical_type = column.logical_type
        semantic_tags = column.semantic_tags
        series = df[column_name]

        # Calculate Aggregation Stats
        if column.is_categorical:
            agg_stats = agg_stats_to_calculate['category']
        elif column.is_numeric:
            agg_stats = agg_stats_to_calculate['numeric']
        elif column.is_datetime:
            agg_stats = agg_stats_to_calculate[Datetime]
        else:
            agg_stats = ["count"]
        values = series.agg(agg_stats).to_dict()

        # Calculate other specific stats based on logical type or semantic tags
        if column.is_boolean:
            values["num_false"] = series.value_counts().get(False, 0)
            values["num_true"] = series.value_counts().get(True, 0)
        elif column.is_numeric:
            float_series = series.astype('float64')  # workaround for https://github.com/pandas-dev/pandas/issues/42626
            quant_values = float_series.quantile([0.25, 0.5, 0.75]).tolist()
            values["first_quartile"] = quant_values[0]
            values["second_quartile"] = quant_values[1]
            values["third_quartile"] = quant_values[2]

        mode = _get_mode(series)
        # The format of the mode should match its format in the DataFrame
        if _is_koalas_dataframe(dataframe) and series.name in latlong_columns:
            mode = list(mode)

        values["nan_count"] = series.isna().sum()
        values["mode"] = mode
        values["physical_type"] = series.dtype
        values["logical_type"] = logical_type
        values["semantic_tags"] = semantic_tags

        # Calculate extra detailed stats, if requested
        if extra_stats:
            if column.is_numeric:
                values["histogram"] = _get_histogram_values(series, bins=bins)
                if isinstance(column.logical_type, (Integer, IntegerNullable)):
                    _range = range(int(values["min"]), int(values["max"]) + 1)
                    # Calculate top numeric values if range of values present
                    # is less than or equal number of histogram bins
                    if len(_range) <= bins:
                        values["top_values"] = _get_numeric_value_counts_in_range(series, _range)
            elif column.is_categorical:
                values["top_values"] = _get_top_values_categorical(series, top_x)
            elif column.is_datetime:
                values["recent_values"] = _get_recent_value_counts(series, recent_x)

        results[column_name] = values
        current_progress = _update_progress(start_time, timer(), 1, current_progress, total_loops, unit, callback)
    return results


def _get_mode(series):
    """Get the mode value for a series"""
    mode_values = series.mode()
    if len(mode_values) > 0:
        return mode_values[0]
    return None


def _replace_nans_for_mutual_info(schema, data):
    """Replace NaN values in the dataframe so that mutual information can be calculated

    Args:
        schema (woodwork.TableSchema): Woodwork typing info for the data
        data (pd.DataFrame): dataframe to use for calculating mutual information

    Returns:
        pd.DataFrame: data with nans replaced with either mean or mode

    """
    for column_name in data.columns[data.isnull().any()]:
        column = schema.columns[column_name]
        series = data[column_name]

        if column.is_numeric or column.is_datetime:
            mean = series.mean()
            if isinstance(mean, float) and not mean.is_integer() and not type(column.logical_type) == Double:
                data[column_name] = series.astype('float')
            data[column_name] = data[column_name].fillna(mean)
        elif column.is_categorical or column.is_boolean:
            mode = _get_mode(series)
            data[column_name] = series.fillna(mode)
    return data


def _make_categorical_for_mutual_info(schema, data, num_bins):
    """Transforms dataframe columns into numeric categories so that
    mutual information can be calculated

    Args:
        schema (woodwork.TableSchema): Woodwork typing info for the data
        data (pd.DataFrame): dataframe to use for calculating mutual information
        num_bins (int): Determines number of bins to use for converting
            numeric features into categorical.

    Returns:
        pd.DataFrame: data with values transformed and binned into numeric categorical values
    """

    for col_name in data.columns:
        column = schema.columns[col_name]
        if column.is_numeric:
            # bin numeric features to make categories
            data[col_name] = pd.qcut(data[col_name], num_bins, duplicates="drop")
        # Convert Datetimes to total seconds - an integer - and bin
        if column.is_datetime:
            data[col_name] = pd.qcut(data[col_name].view('int64'), num_bins, duplicates="drop")
        # convert categories to integers
        new_col = data[col_name]
        if str(new_col.dtype) != 'category':
            new_col = new_col.astype('category')
        data[col_name] = new_col.cat.codes
    return data


def _get_mutual_information_dict(dataframe, num_bins=10, nrows=None, include_index=False, callback=None):
    """Calculates mutual information between all pairs of columns in the DataFrame that
    support mutual information. Logical Types that support mutual information are as
    follows:  Boolean, Categorical, CountryCode, Datetime, Double, Integer, Ordinal,
    PostalCode, and SubRegionCode

    Args:
        dataframe (pd.DataFrame): Data containing Woodwork typing information
            from which to calculate mutual information.
        num_bins (int): Determines number of bins to use for converting
            numeric features into categorical.
        nrows (int): The number of rows to sample for when determining mutual info.
            If specified, samples the desired number of rows from the data.
            Defaults to using all rows.
        include_index (bool): If True, the column specified as the index will be
            included as long as its LogicalType is valid for mutual information calculations.
            If False, the index column will not have mutual information calculated for it.
            Defaults to False.
        callback (callable, optional): function to be called with incremental updates. Has the following parameters:

            - update (int): change in progress since last call
            - progress (int): the progress so far in the calculations
            - total (int): the total number of calculations to do
            - unit (str): unit of measurement for progress/total
            - time_elapsed (float): total time in seconds elapsed since start of call

    Returns:
        list(dict): A list containing dictionaries that have keys `column_1`,
        `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
        Mutual information values are between 0 (no mutual information) and 1
        (perfect dependency).
        """
    start_time = timer()
    unit = 'calculations'
    valid_types = get_valid_mi_types()
    valid_columns = [col_name for col_name, col in dataframe.ww.columns.items() if type(col.logical_type) in valid_types]

    if not include_index and dataframe.ww.index is not None:
        valid_columns.remove(dataframe.ww.index)

    data = dataframe.loc[:, valid_columns]
    if _is_dask_dataframe(data):
        data = data.compute()
    if _is_koalas_dataframe(dataframe):
        data = data.to_pandas()

    # cut off data if necessary
    if nrows is not None and nrows < data.shape[0]:
        data = data.sample(nrows)

    # remove fully null columns
    not_null_cols = data.columns[data.notnull().any()]
    if set(not_null_cols) != set(valid_columns):
        data = data.loc[:, not_null_cols]

    # Setup for progress callback and make initial call
    # Assume 1 unit for preprocessing, n for replace nans, n for make categorical and (n*n+n)/2 for main calculation loop
    n = len(data.columns)
    total_loops = 1 + 2 * n + (n * n + n) / 2
    current_progress = _update_progress(start_time, timer(), 1, 0, total_loops, unit, callback)

    data = _replace_nans_for_mutual_info(dataframe.ww.schema, data)
    current_progress = _update_progress(start_time, timer(), n, current_progress, total_loops, unit, callback)

    data = _make_categorical_for_mutual_info(dataframe.ww.schema, data, num_bins)
    current_progress = _update_progress(start_time, timer(), n, current_progress, total_loops, unit, callback)

    # calculate mutual info for all pairs of columns
    mutual_info = []
    col_names = data.columns.to_list()
    for i, a_col in enumerate(col_names):
        for j in range(i, len(col_names)):
            b_col = col_names[j]
            if a_col == b_col:
                # Ignore because the mutual info for a column with itself will always be 1
                current_progress = _update_progress(start_time, timer(), 1, current_progress, total_loops, unit, callback)
                continue
            else:
                mi_score = normalized_mutual_info_score(data[a_col], data[b_col])
                mutual_info.append(
                    {"column_1": a_col, "column_2": b_col, "mutual_info": mi_score}
                )
                current_progress = _update_progress(start_time, timer(), 1, current_progress, total_loops, unit, callback)
    mutual_info.sort(key=lambda mi: mi['mutual_info'], reverse=True)

    return mutual_info


def _get_value_counts(dataframe, ascending=False, top_n=10, dropna=False):
    """Returns a list of dictionaries with counts for the most frequent values in each column (only
        for columns with `category` as a standard tag).


    Args:
        dataframe (pd.DataFrame, dd.DataFrame, ks.DataFrame): Data from which to count values.
        ascending (bool): Defines whether each list of values should be sorted most frequent
            to least frequent value (False), or least frequent to most frequent value (True).
            Defaults to False.

        top_n (int): the number of top values to retrieve. Defaults to 10.

        dropna (bool): determines whether to remove NaN values when finding frequency. Defaults
            to False.

    Returns:
        list(dict): a list of dictionaries for each categorical column with keys `count`
        and `value`.
    """
    val_counts = {}
    valid_cols = [col for col, column in dataframe.ww.columns.items() if column.is_categorical]
    data = dataframe[valid_cols]
    is_ks = False
    if _is_dask_dataframe(data):
        data = data.compute()
    if _is_koalas_dataframe(data):
        data = data.to_pandas()
        is_ks = True

    for col in valid_cols:
        if dropna and is_ks:
            # Koalas categorical columns will have missing values replaced with the string 'None'
            # Replace them with np.nan so dropna work
            datacol = data[col].replace(to_replace='None', value=np.nan)
        else:
            datacol = data[col]
        frequencies = datacol.value_counts(ascending=ascending, dropna=dropna)
        df = frequencies[:top_n].reset_index()
        df.columns = ["value", "count"]
        values = list(df.to_dict(orient="index").values())
        val_counts[col] = values
    return val_counts


def _get_box_plot_info_for_column(series, quantiles=None):
    """Gets the information necessary to create a box and whisker plot with outliers for a numeric column
        using the IQR method.

    Args:
        series (Series): Data for which the box plot and outlier information will be gathered.
            Will be used to calculate quantiles if none are provided.
        quantiles (dict[float -> float], optional): A dictionary containing the quantiles for the data
            where the key indicates the quantile, and the value is the quantile's value for the data.

    Note:
        The minimum quantiles necessary for outlier detection using the IQR method are the
        first quartile (0.25) and third quartile (0.75). If these keys are missing from the quantiles dictionary,
        the following quantiles will be calculated: {0.0, 0.25, 0.5, 0.75, 1.0}, which correspond to
        {min, first quantile, median, third quantile, max}.

    Returns:
        (dict[str -> float,list[number]]): Returns a dictionary containing box plot information for the Series.
            The following elements will be found in the dictionary:

            - low_bound (float): the lower bound below which outliers lay - to be used as a whisker
            - high_bound (float): the high bound above which outliers lay - to be used as a whisker
            - quantiles (list[float]): the quantiles used to determine the bounds.
                If quantiles were passed in, will contain all quantiles passed in. Otherwise, contains the five
                quantiles {0.0, 0.25, 0.5, 0.75, 1.0}.
            - low_values (list[float, int]): the values of the lower outliers
            - high_values (list[float, int]): the values of the upper outliers
            - low_indices (list[int]): the corresponding index values for each of the lower outliers
            - high_indices (list[int]): the corresponding index values for each of the upper outliers
    """
    if not series.ww._schema.is_numeric:
        raise TypeError('Cannot calculate box plot statistics for non-numeric column')

    if quantiles and not isinstance(quantiles, dict):
        raise TypeError('quantiles must be a dictionary.')

    if dd and isinstance(series, dd.Series):
        series = series.compute()
    if ks and isinstance(series, ks.Series):
        series = series.to_pandas()

    # remove null values from the data
    series = series.dropna()

    # An empty or fully null Series has no outliers, bounds, or quantiles
    if series.shape[0] == 0:
        return {
            'low_bound': np.nan,
            'high_bound': np.nan,
            'quantiles': {0.0: np.nan, 0.25: np.nan, 0.5: np.nan, 0.75: np.nan, 1.0: np.nan},
            "low_values": [],
            "high_values": [],
            "low_indices": [],
            "high_indices": []
        }

    # calculate the outlier bounds using IQR
    if quantiles is None:
        quantiles = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
    elif 0.25 not in quantiles or 0.75 not in quantiles:
        raise ValueError("Input quantiles do not contain the minimum necessary quantiles for outlier calculation: "
                         "0.25 (the first quartile) and 0.75 (the third quartile).")

    q1 = quantiles[0.25]
    q3 = quantiles[0.75]

    iqr = q3 - q1

    low_bound = q1 - (iqr * 1.5)
    high_bound = q3 + (iqr * 1.5)

    # identify outliers in the series
    min = quantiles.get(0.0)
    max = quantiles.get(1.0)
    if min is not None and max is not None and low_bound <= min and high_bound >= max:
        outliers_dict = {
            "low_values": [],
            "high_values": [],
            "low_indices": [],
            "high_indices": []
        }
    else:
        low_series = series[series < low_bound]
        high_series = series[series > high_bound]

        outliers_dict = {
            "low_values": low_series.tolist(),
            "high_values": high_series.tolist(),
            "low_indices": low_series.index.tolist(),
            "high_indices": high_series.index.tolist()
        }

    return {'low_bound': low_bound,
            'high_bound': high_bound,
            'quantiles': quantiles,
            **outliers_dict}


def _get_numeric_value_counts_in_range(series, _range):
    """Count the number of occurrences of integers present in a series with values defined
    by a range of integers. Null values will be ignored.

    Args:
        series (pd.Series): data from which to determine the number of occurrences of values
        _range (type(range)): sequence of integers defining the values for which counts should be made

    Returns:
        value_counts (list(dict)): a list of dictionaries with keys `value` and
            `count`. Output is sorted in descending order based on the value counts.
    """
    frequencies = series.value_counts(dropna=True)
    value_counts = [{"value": i, "count": frequencies[i] if i in frequencies else 0} for i in _range]
    return sorted(value_counts, key=lambda i: (-i["count"], i["value"]))


def _get_top_values_categorical(series, num_x):
    """Get the most frequent values in a pandas Series. Will exclude null values.

    Args:
        column (pd.Series): data to use find most frequent values
        num_x (int): the number of top values to retrieve

    Returns:
        top_list (list(dict)): a list of dictionary with keys `value` and `count`.
            Output is sorted in descending order based on the value counts.
    """
    frequencies = series.value_counts(dropna=True)
    df = frequencies.head(num_x).reset_index()
    df.columns = ["value", "count"]
    df = df.sort_values(["count", "value"], ascending=[False, True])
    value_counts = list(df.to_dict(orient="index").values())
    return value_counts


def _get_recent_value_counts(column, num_x):
    """Get the the number of occurrences of the x most recent values in a datetime column.

    Args:
        column (pd.Series): data to use find value counts
        num_x (int): the number of values to retrieve

    Returns:
        value_counts (list(dict)): a list of dictionary with keys `value` and
            `count`. Output is sorted in descending order based on the value counts.
    """
    datetimes = getattr(column.dt, "date")
    frequencies = datetimes.value_counts(dropna=False)
    values = frequencies.sort_index(ascending=False)[:num_x]
    df = values.reset_index()
    df.columns = ["value", "count"]
    df = df.sort_values(["count", "value"], ascending=[False, True])
    value_counts = list(df.to_dict(orient="index").values())
    return value_counts


def _get_histogram_values(series, bins=10):
    """Get the histogram for a given numeric column.

    Args:
        series (pd.Series): data to use for histogram
        bins (int): the number of bins to use for the histogram

    Returns:
        histogram (list(dict)): a list of dictionary with keys `bins` and
            `frequency`
    """
    values = (
        pd.cut(series, bins=bins, duplicates='drop').value_counts().sort_index()
    )
    df = values.reset_index()
    df.columns = ["bins", "frequency"]
    return list(df.to_dict(orient="index").values())
