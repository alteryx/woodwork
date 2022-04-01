import itertools
import warnings
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_mutual_info_score

from ._bin_numeric_cols_into_categories import _bin_numeric_cols_into_categories

from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.exceptions import ParametersIgnoredWarning, SparseDataWarning
from woodwork.logical_types import IntegerNullable
from woodwork.utils import CallbackCaller, get_valid_mi_types, get_valid_pearson_types


def _get_dependence_dict(
    dataframe,
    measures,
    num_bins=10,
    nrows=None,
    include_index=False,
    callback=None,
    extra_stats=False,
    min_shared=25,
    random_seed=0,
):
    """Calculates dependence measures between all pairs of columns in the DataFrame that
    support measuring dependence. Supports boolean, categorical, datetime, and numeric data.
    Call woodwork.utils.get_valid_mi_types and woodwork.utils.get_valid_pearson_types
    for complete lists of supported Logical Types.

    Args:
        dataframe (pd.DataFrame): Data containing Woodwork typing information
            from which to calculate dependence.
        measures (list or str): Which dependence measures to calculate.
            A list of measures can be provided to calculate multiple
            measures at once.  Valid measure strings:

                - "pearson": calculates the Pearson correlation coefficient
                - "mutual_info": calculates the mutual information between columns
                - "max":  max(abs(pearson), mutual) for each pair of columns
                - "all": includes columns for "pearson", "mutual_info", and "max"
        num_bins (int): Determines number of bins to use for converting numeric
            features into categorical.  Default to 10. Pearson calculation does
            not use binning.
        nrows (int): The number of rows to sample for when determining dependence.
            If specified, samples the desired number of rows from the data.
            Defaults to using all rows.
        include_index (bool): If True, the column specified as the index will be
            included as long as its LogicalType is valid for measuring dependence.
            If False, the index column will not be considered. Defaults to False.
        callback (callable, optional): function to be called with incremental updates. Has the following parameters:

            - update (int): change in progress since last call
            - progress (int): the progress so far in the calculations
            - total (int): the total number of calculations to do
            - unit (str): unit of measurement for progress/total
            - time_elapsed (float): total time in seconds elapsed since start of call
        extra_stats (bool):  If True, additional column "shared_rows"
            recording the number of shared non-null rows for a column
            pair will be included with the dataframe.  If the "max"
            measure is being used, a "measure_used" column will be added
            that records whether Pearson or mutual information was the
            maximum dependence for a particular row. Defaults to False.
        min_shared (int): The number of shared non-null rows needed to
            calculate.  Less rows than this will be considered too sparse
            to measure accurately and will return a NaN value. Must be
            non-negative. Defaults to 25.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    Returns:
        list(dict): A list containing dictionaries that have keys `column_1`,
        `column_2`, and keys for the specified dependence measures. The list is
        sorted in decending order by the first specified measure.
        Dependence information values are between 0 (no dependence) and 1
        (perfect dependency). For Pearson, values range from -1 to 1 but 0 is
        still no dependence.
    """
    start_time = timer()

    measures, calc_order, calc_max = _validate_measures(measures)

    unit = "calculations"
    if calc_order[0] == "pearson":
        pearson_types = get_valid_pearson_types()
        pearson_columns = _get_valid_columns(dataframe, pearson_types)
        valid_columns = pearson_columns
    if "mutual_info" in calc_order:
        mi_types = get_valid_mi_types()
        mutual_columns = _get_valid_columns(dataframe, mi_types)
        # pearson columns are a subset of mutual columns
        valid_columns = mutual_columns

    index = dataframe.ww.index
    if not include_index and index is not None and index in valid_columns:
        valid_columns.remove(index)

    data = dataframe.loc[:, valid_columns]
    # cut off data if necessary
    if _is_dask_dataframe(data):
        data = data.compute()
    elif _is_spark_dataframe(dataframe):
        data = data.to_pandas()
    if nrows is not None and nrows < data.shape[0]:
        data = data.sample(nrows, random_state=random_seed)

    notna_mask = data.notnull()
    not_null_cols = data.columns[notna_mask.any()]
    not_null_col_set = set(not_null_cols)
    if not_null_col_set != set(valid_columns):
        data = data.loc[:, not_null_cols]

    # p: number of pearson columns
    # m: number of mutual columns
    # n: max column size
    if "pearson" in calc_order:
        pearson_columns = [col for col in pearson_columns if col in not_null_col_set]
        p = len(pearson_columns)
    else:
        p = 0
    if "mutual_info" in calc_order:
        mutual_columns = [col for col in mutual_columns if col in not_null_col_set]
        m = len(mutual_columns)
        n = m
    else:
        m = 0
        n = p

    # Setup for progress callback and make initial call
    # Assume 1 unit for preprocessing, n for handling null, m for make categorical
    # combinations in a loop is n! / 2 / (n - 2)! which reduces to (n) (n - 1) / 2
    def _num_calc_steps(n):
        return (n * n - n) / 2

    total_loops = 1 + n + m + _num_calc_steps(p) + _num_calc_steps(m)
    callback_caller = CallbackCaller(callback, unit, total_loops, start_time=start_time)
    callback_caller.update(1)

    # split dataframe into dict of series so we can drop nulls on a per-column basis
    data = {col: data[col].dropna() for col in data}

    # cast nullable type to non-nullable
    for col_name in data:
        column = dataframe.ww.columns[col_name]
        if isinstance(column.logical_type, IntegerNullable):
            cur_dtype = data[col_name].dtype
            data[col_name] = data[col_name].astype(cur_dtype.name.lower())
        if column.is_datetime:
            data[col_name] = data[col_name].view("int64")
    callback_caller.update(n)

    results = defaultdict(dict)

    def _calculate(callback_caller, data, col_names, results, measure):
        column_pairs = itertools.combinations(col_names, 2)
        for a_col, b_col in column_pairs:
            result = results[(a_col, b_col)]
            # check if result already has keys, meaning _calculate has been
            # called previously and some computation can be skipped
            if "column_1" in result:
                num_intersect = result["shared_rows"]
            else:
                result["column_1"] = a_col
                result["column_2"] = b_col
                num_intersect = (notna_mask[a_col] & notna_mask[b_col]).sum()
                result["shared_rows"] = num_intersect

            too_sparse = num_intersect < min_shared
            if too_sparse:
                # TODO: reword since Pearson can return NaN naturally
                warnings.warn(
                    "One or more values in the returned matrix are NaN. A "
                    "NaN value indicates there were not enough rows where "
                    "both columns had non-null data",
                    SparseDataWarning,
                )
                result[measure] = np.nan
            else:
                if "num_union" in result:
                    num_union = result["num_union"]
                else:
                    num_union = (notna_mask[a_col] | notna_mask[b_col]).sum()
                    result["num_union"] = num_union
                intersect = notna_mask[a_col] & notna_mask[b_col]
                if measure == "mutual_info":
                    score = adjusted_mutual_info_score(
                        data[a_col][intersect], data[b_col][intersect]
                    )
                elif measure == "pearson":
                    score = np.corrcoef(data[a_col][intersect], data[b_col][intersect])[
                        0, 1
                    ]

                score = score * num_intersect / num_union
                result[measure] = score
            # increment progress in either case
            callback_caller.update(1)

    for measure in calc_order:
        if measure == "mutual_info":
            data = _bin_numeric_cols_into_categories(
                dataframe.ww.schema, data, num_bins
            )
            callback_caller.update(n)
            col_names = mutual_columns
        elif measure == "pearson":
            col_names = pearson_columns
        _calculate(callback_caller, data, col_names, results, measure)

    for result in results.values():
        if calc_max:
            if "pearson" in result:
                score = pd.Series(
                    [result["mutual_info"], abs(result["pearson"])],
                    index=["mutual_info", "pearson"],
                )
                # to keep sign, get measure of max from index value
                measure_used = score.idxmax()
                # if all measures were nan, measure_used will be nan (float)
                if isinstance(measure_used, float):
                    result["max"] = np.nan
                    if extra_stats:
                        measure_used = "none"
                else:
                    result["max"] = result[measure_used]
                if extra_stats:
                    result["measure_used"] = measure_used
            else:
                result["max"] = result["mutual_info"]
                if extra_stats:
                    result["measure_used"] = "mutual_info"
            if measures == ["max"]:
                # remove results not expected in returned dictionary
                del result["mutual_info"]
                if "pearson" in result:
                    del result["pearson"]
        if "num_union" in result:
            del result["num_union"]
        if not extra_stats:
            del result["shared_rows"]

    results = list(results.values())

    def sort_key(result):
        key = abs(result[measures[0]])
        if np.isnan(key):
            key = -1
        return key

    results.sort(key=sort_key, reverse=True)

    return results


def _get_valid_columns(dataframe, valid_types):
    valid_columns = [
        col_name
        for col_name, col in dataframe.ww.columns.items()
        if type(col.logical_type) in valid_types
    ]
    return valid_columns


def _validate_measures(measures):
    if not isinstance(measures, list):
        if not isinstance(measures, str):
            raise TypeError(f"Supplied measure {measures} is not a string")
        measures = [measures]

    if len(measures) == 0:
        raise ValueError("No measures supplied")

    calc_pearson = False
    calc_mutual = False
    calc_max = False
    calc_order = []
    for measure in measures:
        if measure == "all":
            if not measures == ["all"]:
                warnings.warn(
                    ParametersIgnoredWarning(
                        "additional measures to 'all' measure found; 'all' should be used alone"
                    )
                )
            measures = ["max", "pearson", "mutual_info"]
            calc_pearson = True
            calc_mutual = True
            calc_max = True
            break
        elif measure == "pearson":
            calc_pearson = True
        elif measure == "mutual_info":
            calc_mutual = True
        elif measure == "max":
            calc_pearson = True
            calc_mutual = True
            calc_max = True
        else:
            raise ValueError("Unrecognized dependence measure %s" % measure)

    if calc_pearson:
        calc_order.append("pearson")
    if calc_mutual:
        calc_order.append("mutual_info")

    return measures, calc_order, calc_max
