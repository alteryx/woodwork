from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

from woodwork.logical_types import Datetime, Double, LatLong
from woodwork.utils import _update_progress, get_valid_mi_types, import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def _get_describe_dict(dataframe, include=None, callback=None):
    """Calculates statistics for data contained in a DataFrame using Woodwork typing information.

    Args:
        dataframe (pd.DataFrame): DataFrame to be described with Woodwork typing information initialized
        include (list[str or LogicalType], optional): filter for what columns to include in the
            statistics returned. Can be a list of column names, semantic tags, logical types, or a list
            combining any of the three. It follows the most broad specification. Favors logical types
            then semantic tag then column name. If no matching columns are found, an empty DataFrame
            will be returned.
        callback (callable, optional): function to be called with incremental updates. Has the following parameters:

            - update: percentage change (float between 0 and 100) in progress since last call
            - progress_percent: percentage (float between 0 and 100) of total computation completed
            - time_elapsed: total time in seconds that has elapsed since start of call

    Returns:
        dict[str -> dict]: A dictionary with a key for each column in the data or for each column
        matching the logical types, semantic tags or column names specified in ``include``, paired
        with a value containing a dictionary containing relevant statistics for that column.
    """
    start_time = timer()
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

    if dd and isinstance(dataframe, dd.DataFrame):
        df = dataframe.compute()
    elif ks and isinstance(dataframe, ks.DataFrame):
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
    current_progress = _update_progress(start_time, timer(), 1, 0, total_loops, callback)

    for column_name, column in cols_to_include:
        if 'index' in column.semantic_tags:
            current_progress = _update_progress(start_time, timer(), 1, current_progress, total_loops, callback)
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
            quant_values = series.quantile([0.25, 0.5, 0.75]).tolist()
            values["first_quartile"] = quant_values[0]
            values["second_quartile"] = quant_values[1]
            values["third_quartile"] = quant_values[2]

        mode = _get_mode(series)
        # The format of the mode should match its format in the DataFrame
        if ks and isinstance(dataframe, ks.DataFrame) and series.name in latlong_columns:
            mode = list(mode)

        values["nan_count"] = series.isna().sum()
        values["mode"] = mode
        values["physical_type"] = series.dtype
        values["logical_type"] = logical_type
        values["semantic_tags"] = semantic_tags
        results[column_name] = values
        current_progress = _update_progress(start_time, timer(), 1, current_progress, total_loops, callback)
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
            if isinstance(mean, float) and not type(column.logical_type) == Double:
                data[column_name] = series.astype('float')
            data[column_name] = series.fillna(mean)
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
            data[col_name] = pd.qcut(data[col_name].astype('int64'), num_bins, duplicates="drop")
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

            - update: percentage change (float between 0 and 100) in progress since last call
            - progress_percent: percentage (float between 0 and 100) of total computation completed
            - time_elapsed: total time in seconds that has elapsed since start of call

    Returns:
        list(dict): A list containing dictionaries that have keys `column_1`,
        `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
        Mutual information values are between 0 (no mutual information) and 1
        (perfect dependency).
        """
    start_time = timer()
    valid_types = get_valid_mi_types()
    valid_columns = [col_name for col_name, col in dataframe.ww.columns.items() if type(col.logical_type) in valid_types]

    if not include_index and dataframe.ww.index is not None:
        valid_columns.remove(dataframe.ww.index)

    data = dataframe.loc[:, valid_columns]
    if dd and isinstance(data, dd.DataFrame):
        data = data.compute()
    if ks and isinstance(dataframe, ks.DataFrame):
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
    current_progress = _update_progress(start_time, timer(), 1, 0, total_loops, callback)

    data = _replace_nans_for_mutual_info(dataframe.ww.schema, data)
    current_progress = _update_progress(start_time, timer(), n, current_progress, total_loops, callback)

    data = _make_categorical_for_mutual_info(dataframe.ww.schema, data, num_bins)
    current_progress = _update_progress(start_time, timer(), n, current_progress, total_loops, callback)

    # calculate mutual info for all pairs of columns
    mutual_info = []
    col_names = data.columns.to_list()
    for i, a_col in enumerate(col_names):
        for j in range(i, len(col_names)):
            b_col = col_names[j]
            if a_col == b_col:
                # Ignore because the mutual info for a column with itself will always be 1
                current_progress = _update_progress(start_time, timer(), 1, current_progress, total_loops, callback)
                continue
            else:
                mi_score = normalized_mutual_info_score(data[a_col], data[b_col])
                mutual_info.append(
                    {"column_1": a_col, "column_2": b_col, "mutual_info": mi_score}
                )
                current_progress = _update_progress(start_time, timer(), 1, current_progress, total_loops, callback)
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
    if dd and isinstance(data, dd.DataFrame):
        data = data.compute()
    if ks and isinstance(data, ks.DataFrame):
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
