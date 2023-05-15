import warnings
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np

from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.statistics_utils._bin_numeric_cols_into_categories import (
    _bin_numeric_cols_into_categories,
)
from woodwork.statistics_utils._calculate_dependence_measure import (
    _calculate_dependence_measure,
)
from woodwork.statistics_utils._calculate_max_dependence_for_pair import (
    _calculate_max_dependence_for_pair,
)
from woodwork.statistics_utils._cast_nullable_int_and_datetime_to_int import (
    _cast_nullable_int_and_datetime_to_int,
)
from woodwork.statistics_utils._convert_ordinal_to_numeric import (
    _convert_ordinal_to_numeric,
)
from woodwork.statistics_utils._parse_measures import _parse_measures
from woodwork.utils import (
    CallbackCaller,
    get_valid_mi_types,
    get_valid_pearson_types,
    get_valid_spearman_types,
)


def _get_dependence_dict(
    dataframe,
    measures,
    num_bins=10,
    nrows=None,
    include_index=False,
    include_time_index=False,
    callback=None,
    extra_stats=False,
    min_shared=25,
    random_seed=0,
    max_nunique=6000,
    target_col=None,
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
                - "spearman": calculates the Spearman corerlation coefficient
                - "max":  max(abs(pearson), mutual, abs(spearman)) for each pair of columns
                - "all": includes columns for "pearson", "mutual_info", "spearman", and "max"
        num_bins (int): Determines number of bins to use for converting numeric
            features into categorical.  Default to 10. Pearson calculation does
            not use binning.
        nrows (int): The number of rows to sample for when determining dependence.
            If specified, samples the desired number of rows from the data.
            Defaults to using all rows.
        include_index (bool): If True, the column specified as the index will be
            included as long as its LogicalType is valid for measuring dependence.
            If False, the index column will not be considered. Defaults to False.
        include_time_index (bool): If True, the column specified as the time index will be
            included for measuring dependence.
            If False, the time index column will not be considered. Defaults to False.
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
        max_nunique (int): The total maximum number of unique values for all large categorical columns (> 800 unique values).
            Categorical columns will be dropped until this number is met or until there is only one large categorical column.
            Defaults to 6000.
        target_col (str): The column name of the target. If provided, will only calculate the dependence dictionary between other columns and this target column.
            The target column will be `column_2` in the returned result. Defaults to None.

    Returns:
        list(dict): A list containing dictionaries that have keys `column_1`,
        `column_2`, and keys for the specified dependence measures. The list is
        sorted in decending order by the first specified measure.
        Dependence information values are between 0 (no dependence) and 1
        (perfect dependency). For Pearson, values range from -1 to 1 but 0 is
        still no dependence.
    """
    start_time = timer()
    if target_col is not None and target_col not in list(dataframe.columns):
        raise ValueError("target_col '{}' not in the dataframe".format(target_col))
    boolean_columns = dataframe.ww.select(["Boolean", "BooleanNullable"]).columns.values
    bool_to_int = {col: "IntegerNullable" for col in boolean_columns}
    dataframe_with_bools_to_int = dataframe.ww.copy()
    dataframe_with_bools_to_int.ww.set_types(bool_to_int)
    returned_measures, calc_order, calc_max = _parse_measures(measures)

    unit = "calculations"

    # get valid columns for dependence calculations
    if "pearson" in calc_order:
        pearson_types = get_valid_pearson_types()
        pearson_columns = _get_valid_columns(dataframe_with_bools_to_int, pearson_types)
        valid_columns = pearson_columns
    if "spearman" in calc_order:
        spearman_types = get_valid_spearman_types()
        spearman_columns = _get_valid_columns(
            dataframe_with_bools_to_int,
            spearman_types,
        )
        # pearson columns are a subset of spearman columns
        valid_columns = spearman_columns
    if "mutual_info" in calc_order:
        mi_types = get_valid_mi_types()
        cols_to_drop = _find_large_categorical_columns(
            dataframe_with_bools_to_int,
            max_nunique,
        )
        if len(cols_to_drop):
            warnings.warn(
                "Dropping columns {} to allow mutual information to run faster".format(
                    cols_to_drop,
                ),
                UserWarning,
            )
        mutual_columns = [
            col
            for col in _get_valid_columns(dataframe_with_bools_to_int, mi_types)
            if col not in cols_to_drop
        ]
        # pearson/spearman columns are a subset of mutual columns
        valid_columns = mutual_columns

    index = dataframe_with_bools_to_int.ww.index
    time_index = dataframe_with_bools_to_int.ww.time_index
    if not include_index and index is not None and index in valid_columns:
        valid_columns.remove(index)
    if (
        not include_time_index
        and time_index is not None
        and time_index in valid_columns
    ):
        valid_columns.remove(time_index)

    data = dataframe_with_bools_to_int.loc[:, valid_columns]
    # cut off data if necessary
    if _is_dask_dataframe(data):
        data = data.compute()
    elif _is_spark_dataframe(dataframe_with_bools_to_int):
        data = data.to_pandas()
    if nrows is not None and nrows < data.shape[0]:
        data = data.sample(nrows, random_state=random_seed)

    notna_mask = data.notnull()
    not_null_cols = data.columns[notna_mask.any()]
    not_null_col_set = set(not_null_cols)
    if not_null_col_set != set(valid_columns):
        data = data.loc[:, not_null_cols]

    p = 0  # number of pearson columns
    m = 0  # number of mutual columns
    s = 0  # number of spearman columns
    if "pearson" in calc_order:
        pearson_columns = [col for col in pearson_columns if col in not_null_col_set]
        p = len(pearson_columns)
    if "spearman" in calc_order:
        spearman_columns = [col for col in spearman_columns if col in not_null_col_set]
        s = len(spearman_columns)
    if "mutual_info" in calc_order:
        mutual_columns = [col for col in mutual_columns if col in not_null_col_set]
        m = len(mutual_columns)
    n = max(m, p, s)

    # combinations in a loop is n! / 2 / (n - 2)! which reduces to (n) (n - 1) / 2
    def _num_calc_steps(n):
        return (n * n - n) / 2

    # Assume 1 unit for preprocessing, n for handling nulls, m for binning numerics
    total_loops = (
        1 + n + m + s + _num_calc_steps(p) + _num_calc_steps(m) + _num_calc_steps(s)
    )
    callback_caller = CallbackCaller(callback, unit, total_loops, start_time=start_time)
    callback_caller.update(1)

    # split dataframe into dict of series so we can drop nulls on a per-column basis
    data = {col: data[col].dropna() for col in data}

    # cast nullable type to non-nullable (needed for both pearson and mutual)
    _cast_nullable_int_and_datetime_to_int(data, dataframe_with_bools_to_int.ww.columns)
    callback_caller.update(n)

    results = defaultdict(dict)

    for measure in calc_order:
        if measure == "mutual_info":
            _bin_numeric_cols_into_categories(
                dataframe_with_bools_to_int.ww.schema,
                data,
                num_bins,
            )
            callback_caller.update(n)
            col_names = mutual_columns
        elif measure == "pearson":
            col_names = pearson_columns
        elif measure == "spearman":
            _convert_ordinal_to_numeric(dataframe_with_bools_to_int.ww.schema, data)
            col_names = spearman_columns

        _calculate_dependence_measure(
            measure=measure,
            data=data,
            results=results,
            callback_caller=callback_caller,
            notna_mask=notna_mask,
            min_shared=min_shared,
            col_names=col_names,
            target_col=target_col,
        )

    for result in results.values():
        if calc_max:
            _calculate_max_dependence_for_pair(
                result=result,
                min_shared=min_shared,
                extra_stats=extra_stats,
            )
            if returned_measures == ["max"]:
                # remove measurements not expected in returned dictionary
                del result["mutual_info"]
                if "pearson" in result:
                    del result["pearson"]
                if "spearman" in result:
                    del result["spearman"]

        # Remove cached info not expected in result by user
        if "num_union" in result:
            del result["num_union"]
        if not extra_stats:
            del result["shared_rows"]

    results = list(results.values())

    def sort_key(result):
        key = abs(result[returned_measures[0]])
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


def _find_large_categorical_columns(datatable, total_unique=6000):
    """Finds the categorical columns to drop to speed up mutual information calculations."""

    def categorical_column_drop_helper(df):
        cols_to_drop = []
        cols_greater = df.columns[df.nunique().values > 800]
        if len(cols_greater) < 2:
            return cols_to_drop

        df_uniques = df[cols_greater].nunique()
        total = sum(df_uniques.to_numpy())
        if total > total_unique:
            # try to use mergesort to keep the order of the columns
            if not _is_spark_dataframe(df):
                drop = df_uniques.sort_values(ascending=False, kind="mergesort").index[
                    0
                ]
            else:
                drop = df_uniques.sort_values(ascending=False).index.tolist()[0]
            cols_to_drop.append(drop)
            df = df.drop(cols_to_drop, axis=1)
            cols_to_drop += categorical_column_drop_helper(df)
        return cols_to_drop

    categoricals = datatable.ww.select("category").columns
    # dask dataframe does not have support for `nunique`, but it should be a feature coming in a future release
    if len(categoricals) and not _is_dask_dataframe(datatable):
        return categorical_column_drop_helper(datatable[categoricals])
    return []
