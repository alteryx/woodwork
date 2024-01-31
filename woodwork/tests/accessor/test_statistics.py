import math
import re
import sys
from datetime import datetime
from inspect import isclass
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew

from woodwork.accessor_utils import _is_spark_dataframe, init_series
from woodwork.config import CONFIG_DEFAULTS, config
from woodwork.exceptions import ParametersIgnoredWarning, SparseDataWarning
from woodwork.logical_types import (
    URL,
    Age,
    AgeFractional,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    EmailAddress,
    Filepath,
    Integer,
    IntegerNullable,
    IPAddress,
    LatLong,
    NaturalLanguage,
    Ordinal,
    PersonFullName,
    PhoneNumber,
    PostalCode,
    SubRegionCode,
    Timedelta,
)
from woodwork.statistics_utils import (
    _bin_numeric_cols_into_categories,
    _convert_ordinal_to_numeric,
    _get_describe_dict,
    _get_histogram_values,
    _get_low_high_bound,
    _get_medcouple_statistic,
    _get_mode,
    _get_numeric_value_counts_in_range,
    _get_recent_value_counts,
    _get_top_values_categorical,
)
from woodwork.statistics_utils._get_box_plot_info_for_column import (
    _determine_best_outlier_method,
    _determine_coefficients,
)
from woodwork.statistics_utils._parse_measures import _parse_measures
from woodwork.tests.testing_utils import (
    _check_close,
    check_empty_box_plot_dict,
    concat_dataframe_or_series,
    dep_between_cols,
    to_pandas,
)


def test_get_mode():
    series_list = [
        pd.Series([1, 2, 3, 4, 2, 2, 3]),
        pd.Series(["a", "b", "b", "c", "b"]),
        pd.Series([3, 2, 3, 2]),
        pd.Series([np.nan, np.nan, np.nan]),
        pd.Series([pd.NA, pd.NA, pd.NA]),
        pd.Series([1, 2, np.nan, 2, np.nan, 3, 2]),
        pd.Series([1, 2, pd.NA, 2, pd.NA, 3, 2]),
    ]

    answer_list = [2, "b", 2, None, None, 2, 2]

    for series, answer in zip(series_list, answer_list):
        mode = _get_mode(series)
        if answer is None:
            assert mode is None
        else:
            assert mode == answer


def test_accessor_bin_numeric_cols_into_categories():
    df = pd.DataFrame(
        {
            "ints1": pd.Series([1, 2, 3, 2]),
            "ints2": pd.Series([1, 100, 1, 100]),
            "ints3": pd.Series([1, 2, 3, 2], dtype="Int64"),
            "ints4": pd.Series([0, 1, 1, 0], dtype="Int64"),
            "bools": pd.Series([True, False, True, False]),
            "booleans": pd.Series([True, False, True, False], dtype="boolean"),
            "categories": pd.Series(["test", "test2", "test2", "test"]),
            "dates": pd.Series(
                ["2020-01-01", "2019-01-02", "2020-08-03", "1997-01-04"],
            ),
            "dates2": pd.Series(
                ["2020-01-01", "2020-01-01", "2020-02-01", "2020-01-01"],
            ),
        },
    )
    df.ww.init()
    data = {column: df[column] for column in df.copy()}
    _bin_numeric_cols_into_categories(df.ww.schema, data, num_bins=4)

    assert isinstance(data, dict)

    assert data["ints1"].equals(pd.Series([0, 1, 3, 1], dtype="int8"))
    assert data["ints2"].equals(pd.Series([0, 1, 0, 1], dtype="int8"))
    assert data["ints3"].equals(pd.Series([0, 1, 3, 1], dtype="int8"))
    assert data["ints4"].equals(pd.Series([0, 1, 1, 0], dtype="int8"))
    assert data["bools"].equals(pd.Series([1, 0, 1, 0], dtype="int8"))
    assert data["booleans"].equals(pd.Series([1, 0, 1, 0], dtype="int8"))
    assert data["categories"].equals(pd.Series([0, 1, 1, 0], dtype="int8"))
    assert data["dates"].equals(pd.Series([2, 1, 3, 0], dtype="int8"))
    assert data["dates2"].equals(pd.Series([0, 0, 1, 0], dtype="int8"))


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_same(df_same_mi, measure):
    df_same_mi.ww.init(logical_types={"nans": Categorical(), "ints": "AgeNullable"})
    dep_df = df_same_mi.ww.dependence(measures=measure, min_shared=3)

    cols_used = set(np.unique(dep_df[["column_1", "column_2"]].values))
    assert "nans" not in cols_used
    assert "nat_lang" not in cols_used
    assert dep_df.shape[0] == 1

    if measure == "all":
        measure_columns = ["pearson", "mutual_info", "max"]
    else:
        measure_columns = [measure]
    for measure_col in measure_columns:
        actual = dep_between_cols("floats", "ints", measure_col, dep_df)
        _check_close(actual, 1.0)


@pytest.mark.parametrize(
    "measure",
    ["mutual_info", "pearson", "spearman", "max", "all"],
)
def test_dependence(df_mi, measure):
    df_mi.ww.init(logical_types={"dates": Datetime(datetime_format="%Y-%m-%d")})
    original_df = df_mi.copy()
    dep_df = df_mi.ww.dependence(measures=measure, min_shared=12)
    if measure == "pearson" or measure == "spearman":
        assert dep_df.shape[0] == 6
    else:
        assert dep_df.shape[0] == 15

    if measure == "all":
        measure_columns = ["mutual_info", "max", "pearson", "spearman"]
    else:
        measure_columns = [measure]
    assert sorted(dep_df.columns.tolist()) == sorted(
        ["column_1", "column_2"] + measure_columns,
    )

    if measure == "pearson" or measure == "spearman":
        expected_df = pd.DataFrame(
            data={measure: [0.5, -0.5]},
            index=["dates_ints", "dates_bools"],
        )
    else:
        expected_df = pd.DataFrame(
            data={
                "mutual_info": [1.0, 0.0, 0, 0.208, 0.208],
                "pearson": [-1.0, np.nan, np.nan, 0.5, -0.5],
                "spearman": [-1.0, np.nan, np.nan, 0.5, -0.5],
                "max": [1.0, 0.0, 0, 0.5, -0.5],
            },
            index=[
                "ints_bools",
                "ints_strs",
                "strs_bools",
                "dates_ints",
                "dates_bools",
            ],
        )

    for measurement in measure_columns:
        for row in expected_df.index:
            column_1, column_2 = row.split("_")
            actual = dep_between_cols(column_1, column_2, measurement, dep_df)
            _check_close(actual, expected_df[measurement][row])

    # Confirm that none of this changed the underlying df
    pd.testing.assert_frame_equal(to_pandas(df_mi), to_pandas(original_df))


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_many_rows(df_mi, measure):
    df_mi.ww.init(logical_types={"dates": Datetime(datetime_format="%Y-%m-%d")})
    original_df = df_mi.copy()
    dep_df = df_mi.ww.dependence(measures=measure, min_shared=12)
    many_rows_df = df_mi.ww.dependence(measure, nrows=100000, min_shared=12)
    pd.testing.assert_frame_equal(dep_df, many_rows_df)
    pd.testing.assert_frame_equal(to_pandas(df_mi), to_pandas(original_df))


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_random_seed(df_mi, measure):
    if _is_spark_dataframe(df_mi):
        # TODO: evaluate if koalas order remains same across other machines
        pytest.xfail("koalas sample order differs, may not be deterministic")
    df_mi.ww.init(logical_types={"dates": Datetime(datetime_format="%Y-%m-%d")})
    original_df = df_mi.copy()
    dep_df = df_mi.ww.dependence(measures=measure, nrows=6, min_shared=6, random_seed=2)
    row = dep_df[(dep_df.column_1 == "ints") & (dep_df.column_2 == "dates")].index[0]
    if measure == "all":
        measure = "max"
    if measure == "mutual_info":
        expected = 0.3552453
    else:
        expected = 0.7071067811865474
    np.testing.assert_allclose(dep_df.loc[row][measure], expected)
    pd.testing.assert_frame_equal(to_pandas(df_mi), to_pandas(original_df))


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_one_row(df_mi, measure):
    df_mi.ww.init(logical_types={"dates": Datetime(datetime_format="%Y-%m-%d")})
    original_df = df_mi.copy()
    dep_df = df_mi.ww.dependence(measure, nrows=1, min_shared=1)
    if measure == "pearson":
        assert dep_df.shape[0] == 6
    else:
        assert dep_df.shape[0] == 15
    expected = {"mutual_info": 1.0, "pearson": np.nan, "max": 1.0}
    if measure == "all":
        measure_columns = ["pearson", "mutual_info", "max"]
    else:
        measure_columns = [measure]

    for measure_col in measure_columns:
        for row in dep_df[measure_col]:
            _check_close(row, expected[measure_col])

    pd.testing.assert_frame_equal(to_pandas(df_mi), to_pandas(original_df))


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_num_bins(df_mi, measure):
    df_mi.ww.init(logical_types={"dates": Datetime(datetime_format="%Y-%m-%d")})
    original_df = df_mi.copy()
    dep_df = df_mi.ww.dependence(measure, num_bins=2, min_shared=12)
    if measure == "pearson":
        assert dep_df.shape[0] == 6
    else:
        assert dep_df.shape[0] == 15

    if measure == "pearson":
        expected_df = pd.DataFrame(
            data={"pearson": [0.5, -1.0]},
            index=["dates_ints", "bools_ints"],
        )
    else:
        expected_df = pd.DataFrame(
            data={
                "mutual_info": [1.0, 0.0, 0, 0.208, 1.0],
                "pearson": [-1.0, np.nan, np.nan, 0.5, np.nan],
                "max": [1.0, 0.0, 0, 0.5, 1.0],
            },
            index=[
                "bools_ints",
                "strs_ints",
                "bools_strs",
                "dates_ints",
                "bools_strs2",
            ],
        )

    if measure == "all":
        measure_columns = ["pearson", "mutual_info", "max"]
    else:
        measure_columns = [measure]

    for measurement in measure_columns:
        for row in expected_df.index:
            column_1, column_2 = row.split("_")
            actual = dep_between_cols(column_1, column_2, measurement, dep_df)
            _check_close(actual, expected_df[measurement][row])

    # Confirm that none of this changed the underlying df
    pd.testing.assert_frame_equal(to_pandas(df_mi), to_pandas(original_df))


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_on_index(sample_df, measure):
    sample_df.ww.init(index="id")
    dep_df = sample_df.ww.dependence(measures=measure, min_shared=3)

    assert not ("id" in dep_df["column_1"].values or "id" in dep_df["column_2"].values)

    dep_df = sample_df.ww.dependence(measures=measure, include_index=True)
    assert "id" in dep_df["column_1"].values or "id" in dep_df["column_2"].values


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_on_time_index(sample_df, measure):
    sample_df.ww.init(time_index="signup_date")
    dep_df = sample_df.ww.dependence(measures=measure, min_shared=3)

    assert not (
        "signup_date" in dep_df["column_1"].values
        or "signup_date" in dep_df["column_2"].values
    )

    dep_df = sample_df.ww.dependence(measures=measure, include_time_index=True)
    assert (
        "signup_date" in dep_df["column_1"].values
        or "signup_date" in dep_df["column_2"].values
    )


def test_max_is_nan_extra_stats(sample_df):
    sample_df.ww.init(index="id")
    dep_df = sample_df.ww.dependence(measures="max", min_shared=3, extra_stats=True)
    assert pd.isnull(dep_df["max"][dep_df.index[-1]])


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_returns_empty_df_properly(sample_df, measure):
    schema_df = sample_df[["id", "age"]]
    schema_df.ww.init(index="id")

    dependence_df = schema_df.ww.dependence(measures=measure)
    assert dependence_df.empty


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_sort(df_mi, measure):
    df_mi.ww.init()
    dep_df = df_mi.ww.dependence(measures=measure, min_shared=12)

    if measure == "all":
        measure = "max"

    for i in range(len(dep_df[measure]) - 1):
        current = dep_df[measure].iloc[i]
        next = dep_df[measure].iloc[i + 1]
        if not np.isnan(current):
            if not np.isnan(next):
                assert abs(current) >= abs(next)
        else:
            assert np.isnan(next)


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_dict(df_mi, measure):
    df_mi.ww.init()
    dep_dict = df_mi.ww.dependence_dict(measures=measure, min_shared=12)
    dep_df = df_mi.ww.dependence(measures=measure, min_shared=12)

    pd.testing.assert_frame_equal(pd.DataFrame(dep_dict), dep_df)


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_unique_cols(df_mi_unique, measure):
    df_mi_unique.ww.init()
    dependence_df = df_mi_unique.ww.dependence(measures=measure)

    cols_used = set(np.unique(dependence_df[["column_1", "column_2"]].values))
    if measure != "pearson":
        assert "unique" in cols_used
        assert "unique_with_one_nan" in cols_used
    assert "unique_with_nans" in cols_used
    assert "ints" in cols_used


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_extra_stats(measure):
    df_nans = pd.DataFrame(
        {
            "ints": pd.Series([2, pd.NA, 5, 2], dtype="Int64"),
            "floats": pd.Series([3.3, None, 2.3, 1.3]),
            "bools": pd.Series([True, None, True, False]),
            "bools_pdna": pd.Series([True, pd.NA, True, False], dtype="boolean"),
            "int_to_cat_nan": pd.Series([1, np.nan, 3, 1], dtype="category"),
            "str": pd.Series(["test", np.nan, "test2", "test"]),
            "str_no_nan": pd.Series(["test", "test2", "test2", "test"]),
            "dates": pd.Series(["2020-01-01", None, "2020-01-02", "2020-01-03"]),
        },
    )
    df_nans.ww.init(
        logical_types={
            "str": "Categorical",
            "str_no_nan": "Categorical",
        },
    )
    original_df = df_nans.copy()
    dep_df_extra = df_nans.ww.dependence(measure, extra_stats=True, min_shared=3)
    pd.testing.assert_frame_equal(df_nans, original_df)
    dep_df = df_nans.ww.dependence(measure, min_shared=3)
    pd.testing.assert_frame_equal(dep_df, dep_df_extra[dep_df.columns])

    assert (dep_df_extra["shared_rows"] == 3).all()
    if measure in ("max", "all"):
        assert "measure_used" in dep_df_extra.columns
        # recalculate max to compare
        both_dep_df = df_nans.ww.dependence(
            measures=["mutual_info", "pearson", "spearman"],
            min_shared=3,
        )
        both_dep_df["pearson"] = both_dep_df["pearson"].abs()
        both_dep_df["spearman"] = both_dep_df["spearman"].abs()
        both_dep_df = both_dep_df.set_index(["column_1", "column_2"])
        both_dep_df = both_dep_df.transpose()

        for row in dep_df_extra.index:
            col_1 = dep_df_extra["column_1"][row]
            col_2 = dep_df_extra["column_2"][row]
            expected_max = both_dep_df[col_1][col_2].idxmax()
            assert (
                expected_max == dep_df_extra["measure_used"][row]
                or both_dep_df[col_1][col_2]["pearson"]
                == both_dep_df[col_1][col_2]["mutual_info"]
                or both_dep_df[col_1][col_2]["spearman"]
                == both_dep_df[col_1][col_2]["mutual_info"]
            )
    else:
        assert "measure_used" not in dep_df_extra.columns


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_min_shared(time_index_df, measure):
    time_index_df.ww.init(
        logical_types={"strs": "categorical", "letters": "Categorical"},
    )
    for min_shared in (25, 4, 3):
        dep_df = time_index_df.ww.dependence(measures=measure, min_shared=min_shared)

        if measure == "all":
            measure_columns = ["pearson", "mutual_info", "max"]
        else:
            measure_columns = [measure]

        for measurement in measure_columns:
            if min_shared == 25:
                assert (dep_df[measurement].isna()).all()
            elif min_shared == 4:
                assert not (dep_df[measurement].isna()).all()
                assert (dep_df[measurement].isna()).any()
            elif min_shared == 3:
                if measure == "all" and measurement == "pearson":
                    assert dep_df[measurement].isna().sum() == 9
                else:
                    assert not (dep_df[measurement].isna()).all()
                    assert not (dep_df[measurement].isna()).any()


@pytest.mark.parametrize("measure", ["mutual_info", "pearson", "max", "all"])
def test_dependence_min_shared_warns(time_index_df, measure):
    time_index_df.ww.init(
        logical_types={"strs": "categorical", "letters": "Categorical"},
    )

    msg = (
        "One or more pairs of columns did not share enough rows of non-null "
        "data to measure the relationship.  The measurement for these columns "
        "will be NaN.  Use 'extra_stats=True' to get the shared rows for each "
        "pair of columns."
    )
    with pytest.warns(SparseDataWarning, match=msg):
        time_index_df.ww.dependence(measures=measure, min_shared=25)


@pytest.mark.parametrize(
    "measure, expected",
    [
        ("mutual_info", (18, 7, 28, 28)),
        ("pearson", (8, 5, 11, 11)),
        ("spearman", (8, 5, 15, 11)),
        ("max", (30, 7, 44, 40)),
        ("all", (30, 7, 44, 40)),
    ],
)
def test_dependence_callback(df_mi, measure, expected, mock_callback):
    df_mi.ww.init(logical_types={"dates": Datetime(datetime_format="%Y-%m-%d")})

    df_mi.ww.dependence(measures=measure, callback=mock_callback)

    total_calls, second_call_progress, total, total_progress = expected

    assert len(mock_callback.progress_history) == total_calls

    assert mock_callback.unit == "calculations"
    # First call should be 1 of 26 calculations complete
    assert mock_callback.progress_history[0] == 1
    assert mock_callback.progress_history[1] == second_call_progress

    # Should be 26 calculations at end with a positive elapsed time
    assert mock_callback.total == total
    assert mock_callback.total_update == total_progress
    assert mock_callback.progress_history[-1] == total_progress
    assert mock_callback.total_elapsed_time > 0


@pytest.mark.parametrize(
    "measure",
    ["mutual_info", "pearson", "spearman", "max", "all"],
)
def test_dependence_with_string_index(measure):
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "col1": [1, 2, 3],
            "col2": [10, 20, 30],
        },
    )
    df.ww.init(index="id", logical_types={"id": "unknown"})
    dependence_df = df.ww.dependence(measures=measure)

    cols_used = set(np.unique(dependence_df[["column_1", "column_2"]].values))
    assert "id" not in cols_used
    assert "col1" in cols_used
    assert "col2" in cols_used


def test_dependence_dropna():
    # if regular dropna used, all data will be dropped
    df = pd.DataFrame(
        data={
            "test": [np.nan, 1, 2, 3],
            "case": [0, np.nan, 0, 0],
            "for": [0, 0, np.nan, 0],
            "dropna": [0, 1, 2, np.nan],
        },
    )
    df.ww.init(logical_types={col: Categorical for col in df})
    mi_df = df.ww.mutual_information(min_shared=2)

    expected_df = pd.DataFrame(
        {
            "column_1": {
                0: "case",
                1: "test",
                2: "test",
                3: "test",
                4: "case",
                5: "for",
            },
            "column_2": {
                0: "for",
                1: "case",
                2: "for",
                3: "dropna",
                4: "dropna",
                5: "dropna",
            },
            "mutual_info": {0: 0.5, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        },
    )
    assert mi_df.equals(expected_df)


@pytest.mark.parametrize(
    "logical_types",
    [
        {"a_column": "Unknown"},
        {"a_column": "Categorical"},
        {"a_column": "SubRegionCode"},
    ],
)
def test_dependence_drop_columns(logical_types):
    log_types = {
        "b_column": "Categorical",
        "c_column": "Categorical",
        "d_column": "Categorical",
        **logical_types,
    }
    cat_values = {"a_column": [], "b_column": [], "c_column": [], "d_column": []}
    categoricals_string = "some categorical_{}_{}"
    for k in cat_values.keys():
        num = 3001 if k in ["c_column", "d_column"] else 100
        for n in range(4000):
            # `a_column`, `c_column`, `d_column` all have 3001 unique values, while `b_column` has 100
            cat_values[k].append(categoricals_string.format(k, n % num))

    df = pd.DataFrame(cat_values)
    df.ww.init(logical_types=log_types)
    string_warning = r"Dropping columns \['c_column'\] to allow mutual information"
    with pytest.warns(UserWarning, match=string_warning):
        for dep_dict_str in [
            str(df.ww.dependence_dict()),
            str(df.ww.mutual_information_dict()),
        ]:
            # based on natural column ordering, "c_column" will be missing rather than "d_column"
            # even though both have same number of uniques
            assert "c_column" not in dep_dict_str
            assert "d_column" in dep_dict_str
            if logical_types["a_column"] == "Unknown":
                assert "a_column" not in dep_dict_str
            else:
                assert "a_column" in dep_dict_str


@pytest.mark.parametrize("nunique", [1000, 2001])
def test_dependence_drop_columns_nunique(nunique):
    log_types = {
        "b_column": "Categorical",
        "c_column": "Categorical",
        "d_column": "Categorical",
        "a_column": "Integer",
    }
    cat_values = {
        "a_column": range(1000),
        "b_column": [],
        "c_column": [],
        "d_column": [],
    }
    categoricals_string = "some categorical_{}_{}"
    for k in ["b_column", "c_column", "d_column"]:
        num = 1000 if k in ["c_column", "d_column"] else 100
        for n in range(1000):
            # `a_column`, `c_column`, `d_column` all have 1000 unique values, while `b_column` has 100
            cat_values[k].append(categoricals_string.format(k, n % num))

    df = pd.DataFrame(cat_values)
    df.ww.init(logical_types=log_types)
    for dep_dict_str in [
        str(df.ww.dependence(max_nunique=nunique)),
        str(df.ww.mutual_information(max_nunique=nunique)),
    ]:
        # based on natural column ordering, "c_column" will be missing rather than "d_column"
        # even though both have same number of uniques
        assert "d_column" in dep_dict_str
        assert "a_column" in dep_dict_str
        assert "b_column" in dep_dict_str
        if nunique == 2001:
            assert "c_column" in dep_dict_str
        else:
            assert "c_column" not in dep_dict_str


@pytest.mark.parametrize("df_type", ["pandas", "dask", "spark"])
def test_dependence_drop_columns_dask_spark(df_type):
    log_types = {
        "b_column": "Categorical",
        "c_column": "Categorical",
        "a_column": "Categorical",
    }
    cat_values = {
        "a_column": [],
        "b_column": [],
        "c_column": [],
    }
    categoricals_string = "some categorical_{}_{}"
    for k in cat_values.keys():
        num = 1000 if k in ["a_column", "c_column"] else 100
        for n in range(1000):
            # `a_column`, `c_column`, all have 1000 unique values, while `b_column` has 100
            cat_values[k].append(categoricals_string.format(k, n % num))

    df = pd.DataFrame(cat_values)
    if df_type == "dask":
        dd = pytest.importorskip(
            "dask.dataframe",
            reason="Dask not installed, skipping",
        )
        df = dd.from_pandas(df, npartitions=1)
    elif df_type == "spark":
        ps = pytest.importorskip(
            "pyspark.pandas",
            reason="Spark not installed, skipping",
        )
        df = ps.from_pandas(df)

    df.ww.init(logical_types=log_types)
    for dep_dict_str in [
        str(df.ww.dependence(max_nunique=1000)),
        str(df.ww.mutual_information(max_nunique=1000)),
    ]:
        # based on natural column ordering, "a_column" will be missing rather than "c_column"
        # even though both have same number of uniques
        assert "c_column" in dep_dict_str
        if df_type == "dask":
            assert "a_column" in dep_dict_str
        else:
            assert "a_column" not in dep_dict_str
        assert "b_column" in dep_dict_str


@patch("woodwork.table_accessor._get_dependence_dict")
def test_pearson_dict(_get_dependence_dict, df_mi, mock_callback):
    df_mi.ww.init()
    df_mi.ww.pearson_correlation_dict(
        nrows=100,
        include_index=True,
        include_time_index=True,
        callback=mock_callback,
        extra_stats=True,
        min_shared=25,
        random_seed=5,
    )
    assert _get_dependence_dict.called
    _get_dependence_dict.assert_called_with(
        dataframe=df_mi,
        measures=["pearson"],
        nrows=100,
        include_index=True,
        include_time_index=True,
        callback=mock_callback,
        extra_stats=True,
        min_shared=25,
        random_seed=5,
    )


def test_pearson_method(df_mi, mock_callback):
    df_mi.ww.init()
    with patch.object(df_mi.ww, "pearson_correlation_dict") as pearson_dict_method:
        df_mi.ww.pearson_correlation(
            nrows=100,
            include_index=True,
            include_time_index=True,
            callback=mock_callback,
            extra_stats=True,
            min_shared=25,
            random_seed=5,
        )
    assert pearson_dict_method.called
    pearson_dict_method.assert_called_with(
        nrows=100,
        include_index=True,
        include_time_index=True,
        callback=mock_callback,
        extra_stats=True,
        min_shared=25,
        random_seed=5,
    )


@patch("woodwork.table_accessor._get_dependence_dict")
def test_mutual_dict(_get_dependence_dict, df_mi, mock_callback):
    df_mi.ww.init()
    df_mi.ww.mutual_information_dict(
        num_bins=5,
        nrows=100,
        include_index=True,
        include_time_index=True,
        callback=mock_callback,
        extra_stats=True,
        min_shared=25,
        random_seed=5,
    )
    assert _get_dependence_dict.called
    _get_dependence_dict.assert_called_with(
        dataframe=df_mi,
        measures=["mutual_info"],
        num_bins=5,
        nrows=100,
        include_index=True,
        include_time_index=True,
        callback=mock_callback,
        extra_stats=True,
        min_shared=25,
        random_seed=5,
        max_nunique=6000,
    )


def test_mutual(df_mi, mock_callback):
    df_mi.ww.init()
    with patch.object(df_mi.ww, "mutual_information_dict") as mi_dict_method:
        df_mi.ww.mutual_information(
            num_bins=5,
            nrows=100,
            include_index=True,
            include_time_index=True,
            callback=mock_callback,
            extra_stats=True,
            min_shared=25,
            random_seed=5,
        )
    assert mi_dict_method.called
    mi_dict_method.assert_called_with(
        num_bins=5,
        nrows=100,
        include_index=True,
        include_time_index=True,
        callback=mock_callback,
        extra_stats=True,
        min_shared=25,
        random_seed=5,
        max_nunique=6000,
    )


@patch("woodwork.table_accessor._get_dependence_dict")
def test_spearman_dict(_get_dependence_dict, df_mi, mock_callback):
    df_mi.ww.init()
    df_mi.ww.spearman_correlation_dict(
        nrows=100,
        include_index=True,
        include_time_index=True,
        callback=mock_callback,
        extra_stats=True,
        min_shared=25,
        random_seed=5,
    )
    assert _get_dependence_dict.called
    _get_dependence_dict.assert_called_with(
        dataframe=df_mi,
        measures=["spearman"],
        nrows=100,
        include_index=True,
        include_time_index=True,
        callback=mock_callback,
        extra_stats=True,
        min_shared=25,
        random_seed=5,
    )


def test_spearman(df_mi, mock_callback):
    df_mi.ww.init()
    with patch.object(df_mi.ww, "spearman_correlation_dict") as mi_dict_method:
        df_mi.ww.spearman_correlation(
            nrows=100,
            include_index=True,
            include_time_index=True,
            callback=mock_callback,
            extra_stats=True,
            min_shared=25,
            random_seed=5,
        )
    assert mi_dict_method.called
    mi_dict_method.assert_called_with(
        nrows=100,
        include_index=True,
        include_time_index=True,
        callback=mock_callback,
        extra_stats=True,
        min_shared=25,
        random_seed=5,
    )


def test_get_valid_mi_columns(df_mi):
    df_mi.ww.init()
    valid_columns = df_mi.ww.get_valid_mi_columns()
    mi = df_mi.ww.mutual_information()
    valid_mi_columns = pd.concat([mi.column_1, mi.column_2]).unique()

    assert (valid_columns == valid_mi_columns).all()


def test_get_valid_mi_columns_with_index(sample_df):
    sample_df.ww.init(index="id")
    mi = sample_df.ww.get_valid_mi_columns(include_index=False)
    assert "id" not in mi

    mi = sample_df.ww.get_valid_mi_columns(include_index=True)
    assert "id" in mi


def test_get_valid_mi_columns_with_time_index(sample_df):
    sample_df.ww.init(time_index="signup_date")
    mi = sample_df.ww.get_valid_mi_columns(include_time_index=False)
    assert "signup_date" not in mi

    mi = sample_df.ww.get_valid_mi_columns(include_time_index=True)
    assert "signup_date" in mi


def test_get_describe_dict(describe_df):
    describe_df.ww.init(index="index_col")

    stats_dict = _get_describe_dict(describe_df)
    stats_dict_to_df = pd.DataFrame(stats_dict)
    for extra in ["histogram", "top_values", "recent_values"]:
        assert extra not in stats_dict_to_df.index

    index_order = [
        "physical_type",
        "logical_type",
        "semantic_tags",
        "count",
        "nunique",
        "nan_count",
        "mean",
        "mode",
        "std",
        "min",
        "first_quartile",
        "second_quartile",
        "third_quartile",
        "max",
        "num_true",
        "num_false",
    ]

    stats_dict_to_df = stats_dict_to_df.reindex(index_order)
    stats_df = describe_df.ww.describe()
    pd.testing.assert_frame_equal(stats_df, stats_dict_to_df)


def test_describe_does_not_include_index(describe_df):
    describe_df.ww.init(index="index_col")
    stats_df = describe_df.ww.describe()
    assert "index_col" not in stats_df.columns


def test_describe_accessor_method(describe_df):
    categorical_ltypes = [
        Categorical,
        CountryCode,
        Ordinal(order=("yellow", "red", "blue")),
        PostalCode,
        SubRegionCode,
    ]
    boolean_ltypes = [BooleanNullable]
    non_nullable_boolean_ltypes = [Boolean]
    datetime_ltypes = [Datetime]
    formatted_datetime_ltypes = [Datetime(datetime_format="%Y~%m~%d")]
    timedelta_ltypes = [Timedelta]
    nullable_numeric_ltypes = [Double, IntegerNullable, AgeNullable, AgeFractional]
    non_nullable_numeric_ltypes = [Integer, Age]
    natural_language_ltypes = [
        EmailAddress,
        Filepath,
        PersonFullName,
        IPAddress,
        PhoneNumber,
        URL,
    ]
    latlong_ltypes = [LatLong]

    expected_index = [
        "physical_type",
        "logical_type",
        "semantic_tags",
        "count",
        "nunique",
        "nan_count",
        "mean",
        "mode",
        "std",
        "min",
        "first_quartile",
        "second_quartile",
        "third_quartile",
        "max",
        "num_true",
        "num_false",
    ]

    # Test categorical columns
    category_data = describe_df[["category_col"]]
    if _is_spark_dataframe(category_data):
        expected_dtype = "string"
    else:
        expected_dtype = "category"

    for ltype in categorical_ltypes:
        if isclass(ltype):
            ltype = ltype()

        expected_vals = pd.Series(
            {
                "physical_type": expected_dtype,
                "logical_type": ltype,
                "semantic_tags": {"category", "custom_tag"},
                "count": 7,
                "nunique": 3,
                "nan_count": 1,
                "mode": "red",
            },
            name="category_col",
        )
        category_data.ww.init(
            logical_types={"category_col": ltype},
            semantic_tags={"category_col": "custom_tag"},
        )
        stats_df = category_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"category_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["category_col"].dropna())

    # Test nullable boolean columns
    boolean_data = describe_df[["boolean_col"]]
    for ltype in boolean_ltypes:
        expected_dtype = ltype.primary_dtype
        expected_vals = pd.Series(
            {
                "physical_type": expected_dtype,
                "logical_type": ltype(),
                "semantic_tags": {"custom_tag"},
                "count": 7,
                "nan_count": 1,
                "mode": True,
                "num_true": 4,
                "num_false": 3,
            },
            name="boolean_col",
        )
        boolean_data.ww.init(
            logical_types={"boolean_col": ltype},
            semantic_tags={"boolean_col": "custom_tag"},
        )
        stats_df = boolean_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"boolean_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["boolean_col"].dropna())

    # Test non-nullable boolean columns
    boolean_data = describe_df[["boolean_col"]].fillna(True)
    for ltype in non_nullable_boolean_ltypes:
        expected_dtype = ltype.primary_dtype
        expected_vals = pd.Series(
            {
                "physical_type": expected_dtype,
                "logical_type": ltype(),
                "semantic_tags": {"custom_tag"},
                "count": 8,
                "nan_count": 0,
                "mode": True,
                "num_true": 5,
                "num_false": 3,
            },
            name="boolean_col",
        )
        boolean_data.ww.init(
            logical_types={"boolean_col": ltype},
            semantic_tags={"boolean_col": "custom_tag"},
        )
        stats_df = boolean_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"boolean_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["boolean_col"].dropna())

    # Test datetime columns
    datetime_data = describe_df[["datetime_col"]]
    for ltype in datetime_ltypes:
        expected_vals = pd.Series(
            {
                "physical_type": ltype.primary_dtype,
                "logical_type": ltype(),
                "semantic_tags": {"custom_tag"},
                "count": 7,
                "nunique": 6,
                "nan_count": 1,
                "mean": pd.Timestamp("2020-01-19 09:25:42.857142784"),
                "mode": pd.Timestamp("2020-02-01 00:00:00"),
                "min": pd.Timestamp("2020-01-01 00:00:00"),
                "max": pd.Timestamp("2020-02-02 18:00:00"),
            },
            name="datetime_col",
        )
        datetime_data.ww.init(
            logical_types={"datetime_col": ltype},
            semantic_tags={"datetime_col": "custom_tag"},
        )
        stats_df = datetime_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"datetime_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["datetime_col"].dropna())

    # Test formatted datetime columns
    formatted_datetime_data = describe_df[["formatted_datetime_col"]]
    for ltype in formatted_datetime_ltypes:
        converted_to_datetime = pd.to_datetime(
            [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
                "2020-02-02",
                "2020-03-02",
                pd.NaT,
                "2020-02-01",
                "2020-01-02",
            ],
        )
        expected_vals = pd.Series(
            {
                "physical_type": ltype.primary_dtype,
                "logical_type": ltype,
                "semantic_tags": {"custom_tag"},
                "count": 7,
                "nunique": 6,
                "nan_count": 1,
                "mean": converted_to_datetime.mean(),
                "mode": pd.to_datetime("2020-02-01"),
                "min": converted_to_datetime.min(),
                "max": converted_to_datetime.max(),
            },
            name="formatted_datetime_col",
        )
        formatted_datetime_data.ww.init(
            logical_types={"formatted_datetime_col": ltype},
            semantic_tags={"formatted_datetime_col": "custom_tag"},
        )
        stats_df = formatted_datetime_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"formatted_datetime_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["formatted_datetime_col"].dropna())

    # Test timedelta columns - Skip for Spark
    if not _is_spark_dataframe(describe_df):
        timedelta_data = describe_df["timedelta_col"]
        for ltype in timedelta_ltypes:
            expected_vals = pd.Series(
                {
                    "physical_type": ltype.primary_dtype,
                    "logical_type": ltype(),
                    "semantic_tags": {"custom_tag"},
                    "count": 7,
                    "nan_count": 1,
                    "mode": pd.Timedelta("31days"),
                },
                name="col",
            )
            df = pd.DataFrame({"col": timedelta_data})
            df.ww.init(
                logical_types={"col": ltype},
                semantic_tags={"col": "custom_tag"},
            )
            stats_df = df.ww.describe()
            assert isinstance(stats_df, pd.DataFrame)
            assert set(stats_df.columns) == {"col"}
            assert stats_df.index.tolist() == expected_index
            assert expected_vals.equals(stats_df["col"].dropna())

    # Test numeric columns with nullable ltypes
    numeric_data = describe_df[["numeric_col"]]
    for ltype in nullable_numeric_ltypes:
        expected_vals = pd.Series(
            {
                "physical_type": ltype.primary_dtype,
                "logical_type": ltype(),
                "semantic_tags": {"numeric", "custom_tag"},
                "count": 7,
                "nunique": 6,
                "nan_count": 1,
                "mean": 20.857142857142858,
                "mode": 10,
                "std": 18.27957486220227,
                "min": 1,
                "first_quartile": 10,
                "second_quartile": 17,
                "third_quartile": 26,
                "max": 56,
            },
            name="numeric_col",
        )
        numeric_data.ww.init(
            logical_types={"numeric_col": ltype},
            semantic_tags={"numeric_col": "custom_tag"},
        )
        stats_df = numeric_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"numeric_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["numeric_col"].dropna())

    # Test numeric with non-nullable ltypes
    numeric_data = describe_df[["numeric_col"]].dropna()
    for ltype in nullable_numeric_ltypes:
        expected_vals = pd.Series(
            {
                "physical_type": ltype.primary_dtype,
                "logical_type": ltype(),
                "semantic_tags": {"numeric", "custom_tag"},
                "count": 7,
                "nunique": 6,
                "nan_count": 0,
                "mean": 20.857142857142858,
                "mode": 10,
                "std": 18.27957486220227,
                "min": 1,
                "first_quartile": 10,
                "second_quartile": 17,
                "third_quartile": 26,
                "max": 56,
            },
            name="numeric_col",
        )
        numeric_data.ww.init(
            logical_types={"numeric_col": ltype},
            semantic_tags={"numeric_col": "custom_tag"},
        )
        stats_df = numeric_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"numeric_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["numeric_col"].dropna())

    numeric_data = describe_df[["numeric_col"]].fillna(0)
    for ltype in non_nullable_numeric_ltypes:
        expected_vals = pd.Series(
            {
                "physical_type": ltype.primary_dtype,
                "logical_type": ltype(),
                "semantic_tags": {"numeric", "custom_tag"},
                "count": 8,
                "nunique": 7,
                "nan_count": 0,
                "mean": 18.25,
                "mode": 10,
                "std": 18.460382289804137,
                "min": 0,
                "first_quartile": 7.75,
                "second_quartile": 13.5,
                "third_quartile": 23,
                "max": 56,
            },
            name="numeric_col",
        )
        numeric_data.ww.init(
            logical_types={"numeric_col": ltype},
            semantic_tags={"numeric_col": "custom_tag"},
        )
        stats_df = numeric_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"numeric_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["numeric_col"].dropna())

    # Test natural language columns
    natural_language_data = describe_df[["natural_language_col"]]
    expected_dtype = "string"
    for ltype in natural_language_ltypes:
        expected_vals = pd.Series(
            {
                "physical_type": expected_dtype,
                "logical_type": ltype(),
                "semantic_tags": {"custom_tag"},
                "count": 7,
                "nan_count": 1,
                "mode": "Duplicate sentence.",
            },
            name="natural_language_col",
        )
        natural_language_data.ww.init(
            logical_types={"natural_language_col": ltype},
            semantic_tags={"natural_language_col": "custom_tag"},
        )
        stats_df = natural_language_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"natural_language_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["natural_language_col"].dropna())

    # Test latlong columns
    latlong_data = describe_df[["latlong_col"]]
    expected_dtype = "object"
    for ltype in latlong_ltypes:
        mode = [0, 0] if _is_spark_dataframe(describe_df) else (0, 0)
        expected_vals = pd.Series(
            {
                "physical_type": expected_dtype,
                "logical_type": ltype(),
                "semantic_tags": {"custom_tag"},
                "count": 6,
                "nan_count": 2,
                "mode": mode,
            },
            name="latlong_col",
        )
        latlong_data.ww.init(
            logical_types={"latlong_col": ltype},
            semantic_tags={"latlong_col": "custom_tag"},
        )
        stats_df = latlong_data.ww.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {"latlong_col"}
        assert stats_df.index.tolist() == expected_index
        assert expected_vals.equals(stats_df["latlong_col"].dropna())


@patch.object(sys.modules["woodwork.statistics_utils._get_describe_dict"], "percentile")
@pytest.mark.parametrize(
    "nullable_numeric_type",
    [Double, IntegerNullable, AgeNullable, AgeFractional],
)
def test_percentile_func_not_called_with_nans(
    mock_percentile,
    describe_df,
    nullable_numeric_type,
):
    numeric_data = describe_df[["numeric_col"]]
    numeric_data.ww.init(
        logical_types={"numeric_col": nullable_numeric_type},
        semantic_tags={"numeric_col": "custom_tag"},
    )
    numeric_data.ww.describe()
    assert not mock_percentile.called


@patch.object(sys.modules["woodwork.statistics_utils._get_describe_dict"], "percentile")
@pytest.mark.parametrize("non_nullable_numeric_type", [Integer, Age])
def test_percentile_func_called_without_nans(
    mock_percentile,
    describe_df,
    non_nullable_numeric_type,
):
    numeric_data = describe_df[["numeric_col"]].fillna(0)
    numeric_data.ww.init(
        logical_types={"numeric_col": non_nullable_numeric_type},
        semantic_tags={"numeric_col": "custom_tag"},
    )
    numeric_data.ww.describe()
    assert mock_percentile.called


def test_describe_with_improper_tags(describe_df):
    df = describe_df.copy()[["boolean_col", "natural_language_col"]]

    logical_types = {
        "boolean_col": BooleanNullable,
        "natural_language_col": NaturalLanguage,
    }
    semantic_tags = {
        "boolean_col": "category",
        "natural_language_col": "numeric",
    }

    df.ww.init(logical_types=logical_types, semantic_tags=semantic_tags)
    stats_df = df.ww.describe()

    # Make sure boolean stats were computed with improper 'category' tag
    assert isinstance(stats_df["boolean_col"]["logical_type"], BooleanNullable)
    assert stats_df["boolean_col"]["semantic_tags"] == {"category"}
    # Make sure numeric stats were not computed with improper 'numeric' tag
    assert stats_df["natural_language_col"]["semantic_tags"] == {"numeric"}
    assert (
        stats_df["natural_language_col"][["mean", "std", "min", "max"]].isnull().all()
    )


def test_describe_with_no_semantic_tags(describe_df):
    df = describe_df.copy()[["category_col", "numeric_col"]]

    logical_types = {
        "category_col": Categorical,
        "numeric_col": IntegerNullable,
    }

    df.ww.init(logical_types=logical_types, use_standard_tags=False)
    stats_df = df.ww.describe()
    assert df.ww.semantic_tags["category_col"] == set()
    assert df.ww.semantic_tags["numeric_col"] == set()

    # Make sure category stats were computed
    assert stats_df["category_col"]["semantic_tags"] == set()
    assert stats_df["category_col"]["nunique"] == 3
    # Make sure numeric stats were computed
    assert stats_df["numeric_col"]["semantic_tags"] == set()
    np.testing.assert_almost_equal(stats_df["numeric_col"]["mean"], 20.85714, 5)


def test_describe_with_include(sample_df):
    semantic_tags = {"full_name": "tag1", "email": ["tag2"], "age": ["numeric", "age"]}
    sample_df.ww.init(semantic_tags=semantic_tags)

    col_name_df = sample_df.ww.describe(include=["full_name"])
    assert col_name_df.shape == (16, 1)
    assert "full_name", "email" in col_name_df.columns

    semantic_tags_df = sample_df.ww.describe(["tag1", "tag2"])
    assert "full_name" in col_name_df.columns
    assert len(semantic_tags_df.columns) == 2

    logical_types_df = sample_df.ww.describe([Datetime, BooleanNullable])
    assert set(logical_types_df.columns) == {
        "signup_date",
        "is_registered",
        "datetime_with_NaT",
    }

    multi_params_df = sample_df.ww.describe(["age", "tag1", Datetime])
    expected = ["full_name", "age", "signup_date", "datetime_with_NaT"]
    for col_name in expected:
        assert col_name in multi_params_df.columns
    multi_params_df["full_name"].equals(col_name_df["full_name"])
    multi_params_df["full_name"].equals(sample_df.ww.describe()["full_name"])


def test_describe_numeric_all_nans():
    df = pd.DataFrame({"nulls": [np.nan] * 5})
    logical_types = ["double", "integer_nullable"]

    for logical_type in logical_types:
        df.ww.init(logical_types={"nulls": logical_type})
        stats = df.ww.describe_dict(extra_stats=True)
        assert pd.isnull(stats["nulls"]["max"])
        assert pd.isnull(stats["nulls"]["min"])
        assert pd.isnull(stats["nulls"]["mean"])
        assert pd.isnull(stats["nulls"]["std"])
        assert stats["nulls"]["nan_count"] == 5
        assert stats["nulls"]["histogram"] == []
        assert stats["nulls"]["top_values"] == []


def test_describe_with_no_match(sample_df):
    sample_df.ww.init()
    df = sample_df.ww.describe(include=["wrongname"])
    assert df.empty


def test_describe_add_result_callback(describe_df, mock_results_callback):
    describe_df.ww.init(index="index_col")

    description = describe_df.ww.describe(results_callback=mock_results_callback)
    index_order = [
        "physical_type",
        "logical_type",
        "semantic_tags",
        "count",
        "nunique",
        "nan_count",
        "mean",
        "mode",
        "std",
        "min",
        "first_quartile",
        "second_quartile",
        "third_quartile",
        "max",
        "num_true",
        "num_false",
    ]
    all_results = mock_results_callback.results_so_far
    actual_results = all_results[-1]
    actual_results = actual_results.reindex(index_order)
    actual_most_recent = mock_results_callback.most_recent_calculation

    pd.testing.assert_frame_equal(actual_results, description)
    for ind, new_updated in enumerate(all_results):
        assert new_updated.shape[1] == ind + 1
    # Spark df does not have timedelta column
    if _is_spark_dataframe(describe_df):
        assert len(all_results) == 8
    else:
        assert len(all_results) == 9
    assert actual_most_recent[0].name == "boolean_col"
    assert actual_most_recent[-1].name == "unknown_col"


def test_describe_callback(describe_df, mock_callback):
    describe_df.ww.init(index="index_col")

    describe_df.ww.describe(callback=mock_callback)

    assert mock_callback.unit == "calculations"
    # Spark df does not have timedelta column
    if _is_spark_dataframe(describe_df):
        ncalls = 10
    else:
        ncalls = 11

    assert len(mock_callback.progress_history) == ncalls

    # First call should be 1 unit complete
    assert mock_callback.progress_history[0] == 1
    # After second call should be 2 unit complete
    assert mock_callback.progress_history[1] == 2

    # Should be ncalls at end with a positive elapsed time
    assert mock_callback.total == ncalls
    assert mock_callback.total_update == ncalls
    assert mock_callback.progress_history[-1] == ncalls
    assert mock_callback.total_elapsed_time > 0


@pytest.mark.parametrize("use_age", [True, False])
def test_describe_dict_extra_stats(use_age, describe_df):
    describe_df = describe_df.drop(
        columns=[
            "boolean_col",
            "natural_language_col",
            "formatted_datetime_col",
            "timedelta_col",
            "latlong_col",
        ],
    )
    describe_df["nullable_integer_col"] = describe_df["numeric_col"]
    describe_df["integer_col"] = describe_df["numeric_col"].fillna(0)
    describe_df["small_range_col"] = describe_df["numeric_col"].fillna(0) // 10
    describe_df["small_range_col_ints_as_double"] = (
        describe_df["numeric_col"].fillna(0) // 10.0
    )
    describe_df["small_range_col_double_not_valid"] = (
        describe_df["numeric_col"].fillna(0) / 10
    )

    ltypes = {
        "category_col": "Categorical",
        "datetime_col": "Datetime",
        "numeric_col": "Double",
        "nullable_integer_col": "IntegerNullable",
        "integer_col": "Integer",
        "small_range_col": "Integer",
        "small_range_col_ints_as_double": "Double",
        "small_range_col_double_not_valid": "Double",
    }
    if use_age:
        ltypes.update(
            {
                "numeric_col": "AgeFractional",
                "nullable_integer_col": "AgeNullable",
                "integer_col": "Age",
                "small_range_col": "Age",
                "small_range_col_ints_as_double": "AgeFractional",
                "small_range_col_double_not_valid": "AgeFractional",
            },
        )
    describe_df.ww.init(index="index_col", logical_types=ltypes)
    desc_dict = describe_df.ww.describe_dict(extra_stats=True)

    # category columns should have top_values
    assert isinstance(desc_dict["category_col"]["top_values"], list)
    assert desc_dict["category_col"].get("histogram") is None
    assert desc_dict["category_col"].get("recent_values") is None

    # datetime columns should have recent_values
    assert isinstance(desc_dict["datetime_col"]["recent_values"], list)
    assert desc_dict["datetime_col"].get("histogram") is None
    assert desc_dict["datetime_col"].get("top_values") is None

    # numeric columns should have histogram
    for col in [
        "numeric_col",
        "nullable_integer_col",
        "integer_col",
        "small_range_col",
        "small_range_col_ints_as_double",
        "small_range_col_double_not_valid",
    ]:
        assert isinstance(desc_dict[col]["histogram"], list)
        assert desc_dict[col].get("recent_values") is None
        if col in {"small_range_col"}:
            # If values are in a narrow range, top values should be present
            assert isinstance(desc_dict[col]["top_values"], list)
        else:
            assert desc_dict[col].get("top_values") is None


@patch.object(
    sys.modules["woodwork.statistics_utils._get_describe_dict"],
    "_get_numeric_value_counts_in_range",
)
def test_describe_dict_extra_stats_overflow_range(
    mock_get_numeric_value_counts_in_range,
    describe_df,
):
    df = pd.DataFrame(
        {
            "large_range": [-9215883799005046784, 0, 1, 9223177510267041793],
            "large_nums": [97896598486960007123867158471523621205853924, 0, 1, 2],
        },
    )
    df.ww.init()

    assert not mock_get_numeric_value_counts_in_range.called

    # Confirm we don't make it inside the block that calls _get_numeric_value_counts_in_range,
    # since that would still have an overflow error. This is okay, because when the range is so
    # large that it causes overflow errors, it'd be absurd for there to be more bins such that
    # we should go into that block of code
    df.ww.describe_dict(extra_stats=True)
    assert not mock_get_numeric_value_counts_in_range.called

    # Confirm we actually have the ability to make it into that block
    # by shrinking the range and keeping the integer values
    describe_df["small_range_col"] = (
        describe_df["numeric_col"].fillna(0) // 10
    ).astype("Int64")
    describe_df.ww.init(index="index_col")
    describe_df.ww.describe_dict(extra_stats=True)
    assert mock_get_numeric_value_counts_in_range.called


def test_value_counts(categorical_df):
    nan = np.nan
    logical_types = {
        "ints": IntegerNullable,
        "categories1": Categorical,
        "bools": BooleanNullable,
        "categories2": Categorical,
        "categories3": Categorical,
    }
    categorical_df.ww.init(logical_types=logical_types)
    val_cts = categorical_df.ww.value_counts()
    for col in categorical_df.ww.columns:
        if col in ["ints", "bools"]:
            assert col not in val_cts
        else:
            assert col in val_cts

    expected_cat1 = [
        {"value": 200, "count": 4},
        {"value": 100, "count": 3},
        {"value": 1, "count": 2},
        {"value": 3, "count": 1},
    ]
    # Spark converts numeric categories to strings, so we need to update the expected values for this
    # Spark will result in `pd.NA` instead of `np.nan` in categorical columns
    if _is_spark_dataframe(categorical_df):
        updated_results = []
        for items in expected_cat1:
            updated_results.append(
                {k: (str(v) if k == "value" else v) for k, v in items.items()},
            )
        expected_cat1 = updated_results

        nan = None

    expected_cat2 = [
        {"value": nan, "count": 6},
        {"value": "test", "count": 3},
        {"value": "test2", "count": 1},
    ]
    expected_cat3 = [
        {"value": nan, "count": 7},
        {"value": "test", "count": 3},
    ]

    assert val_cts["categories1"] == expected_cat1
    assert val_cts["categories2"] == expected_cat2
    assert val_cts["categories3"] == expected_cat3

    val_cts_descending = categorical_df.ww.value_counts(ascending=True)
    for col, vals in val_cts_descending.items():
        for i in range(len(vals)):
            assert vals[i]["count"] == val_cts[col][-i - 1]["count"]

    val_cts_dropna = categorical_df.ww.value_counts(dropna=True)
    assert val_cts_dropna["categories3"] == [{"value": "test", "count": 3}]

    val_cts_2 = categorical_df.ww.value_counts(top_n=2)
    for col in val_cts_2:
        assert len(val_cts_2[col]) == 2


def test_datetime_get_recent_value_counts():
    times = pd.Series(
        [
            datetime(2019, 2, 2, 1, 10, 0, 1),
            datetime(2019, 4, 2, 2, 20, 1, 0),
            datetime(2019, 3, 1, 3, 30, 1, 0),
            datetime(2019, 5, 1, 4, 40, 1, 0),
            datetime(2019, 1, 1, 5, 50, 1, 0),
            datetime(2019, 4, 2, 6, 10, 1, 0),
            datetime(2019, 4, 2, 7, 20, 1, 0),
            datetime(2019, 5, 1, 8, 30, 0, 0),
            pd.NaT,
        ],
    )
    values = _get_recent_value_counts(times, num_x=3)
    expected_values = [
        {"value": datetime(2019, 4, 2).date(), "count": 3},
        {"value": datetime(2019, 5, 1).date(), "count": 2},
        {"value": datetime(2019, 3, 1).date(), "count": 1},
    ]
    assert values == expected_values


def test_numeric_histogram():
    column = pd.Series(np.random.randn(1000))
    column = pd.concat([column, pd.Series([np.nan])])
    bins = 7
    values = _get_histogram_values(column, bins=bins)
    assert len(values) == bins
    total = 0
    for info in values:
        assert "bins" in info
        assert len(info["bins"]) == 2
        assert isinstance(info["bins"][0], float)
        assert isinstance(info["bins"][1], float)
        assert "frequency" in info
        freq = info["frequency"]
        total += freq
    assert total == 1000


@pytest.mark.parametrize(
    "input_series, expected",
    [
        (
            ["a", "b", "b", "c", "c", "c", np.nan],
            [
                {"value": "c", "count": 3},
                {"value": "b", "count": 2},
                {"value": "a", "count": 1},
            ],
        ),
        (
            [1, 2, 2, 3],
            [
                {"value": 2, "count": 2},
                {"value": 1, "count": 1},
                {"value": 3, "count": 1},
            ],
        ),
    ],
)
def test_get_top_values_categorical(input_series, expected):
    column = pd.Series(input_series)
    top_values = _get_top_values_categorical(column, 10)
    assert top_values == expected


@pytest.mark.parametrize(
    "input_series, expected",
    [
        (
            pd.Series([1, 2, 2, 3, 3, 3, pd.NA], dtype="Int64"),
            [
                {"value": 3, "count": 3},
                {"value": 2, "count": 2},
                {"value": 1, "count": 1},
                {"value": 0, "count": 0},
            ],
        ),
        (
            pd.Series([1, 2, 2, 3], dtype="int64"),
            [
                {"value": 2, "count": 2},
                {"value": 1, "count": 1},
                {"value": 3, "count": 1},
                {"value": 0, "count": 0},
            ],
        ),
    ],
)
def test_get_numeric_value_counts_in_range(input_series, expected):
    column = input_series
    top_values = _get_numeric_value_counts_in_range(column, range(4))
    assert top_values == expected


def test_box_plot_outliers(outliers_df):
    outliers_series = outliers_df["has_outliers"]
    outliers_series.ww.init()

    no_outliers_series = outliers_df["no_outliers"]
    no_outliers_series.ww.init()

    has_outliers_dict = outliers_series.ww.box_plot_dict()
    assert has_outliers_dict["low_bound"] == 8.125
    assert has_outliers_dict["high_bound"] == 83.125
    assert has_outliers_dict["quantiles"] == {
        0.0: -16.0,
        0.25: 36.25,
        0.5: 42.0,
        0.75: 55.0,
        1.0: 93.0,
    }
    assert has_outliers_dict["low_values"] == [-16]
    assert has_outliers_dict["high_values"] == [93]
    assert has_outliers_dict["low_indices"] == [3]
    assert has_outliers_dict["high_indices"] == [0]

    no_outliers_dict = no_outliers_series.ww.box_plot_dict()

    # Since there are no outliers, the the low bound is the min value
    # and the high bound is the max value
    assert no_outliers_dict["low_bound"] == 23.0
    assert no_outliers_dict["high_bound"] == 60.0
    assert no_outliers_dict["quantiles"] == {
        0.0: 23.0,
        0.25: 36.25,
        0.5: 42.0,
        0.75: 55.0,
        1.0: 60.0,
    }
    assert no_outliers_dict["low_values"] == []
    assert no_outliers_dict["high_values"] == []
    assert no_outliers_dict["low_indices"] == []
    assert no_outliers_dict["high_indices"] == []


def test_box_plot_outliers_with_quantiles(outliers_df):
    outliers_series = outliers_df["has_outliers"]
    outliers_series.ww.init()

    no_outliers_series = outliers_df["no_outliers"]
    no_outliers_series.ww.init()

    has_outliers_dict = outliers_series.ww.box_plot_dict(
        quantiles={0.0: -16.0, 0.25: 36.25, 0.5: 42.0, 0.75: 55.0, 1.0: 93.0},
    )
    assert has_outliers_dict["method"] == "box_plot"
    assert has_outliers_dict["low_bound"] == 8.125
    assert has_outliers_dict["high_bound"] == 83.125
    assert has_outliers_dict["quantiles"] == {
        0.0: -16.0,
        0.25: 36.25,
        0.5: 42.0,
        0.75: 55.0,
        1.0: 93.0,
    }
    assert has_outliers_dict["low_values"] == [-16]
    assert has_outliers_dict["high_values"] == [93]
    assert has_outliers_dict["low_indices"] == [3]
    assert has_outliers_dict["high_indices"] == [0]

    no_outliers_dict = no_outliers_series.ww.medcouple_dict(
        quantiles={0.0: 23.0, 0.25: 36.25, 0.5: 42.0, 0.75: 55.0, 1.0: 60.0},
    )
    assert no_outliers_dict["method"] == "medcouple"
    assert no_outliers_dict["low_bound"] == 23.0
    assert no_outliers_dict["high_bound"] == 60.0
    assert no_outliers_dict["quantiles"] == {
        0.0: 23.0,
        0.25: 36.25,
        0.5: 42.0,
        0.75: 55.0,
        1.0: 60.0,
    }
    assert no_outliers_dict["low_values"] == []
    assert no_outliers_dict["high_values"] == []
    assert no_outliers_dict["low_indices"] == []
    assert no_outliers_dict["high_indices"] == []


def test_get_outliers_for_column_with_nans_box_plot(outliers_df):
    contains_nans_series = outliers_df["has_outliers_with_nans"]
    contains_nans_series.ww.init()

    box_plot_dict = contains_nans_series.ww.box_plot_dict()
    assert box_plot_dict["method"] == "box_plot"
    assert box_plot_dict["low_bound"] == 4.5
    assert box_plot_dict["high_bound"] == 88.5
    assert box_plot_dict["quantiles"] == {
        0.0: -16.0,
        0.25: 36.0,
        0.5: 42.0,
        0.75: 57.0,
        1.0: 93.0,
    }
    assert box_plot_dict["low_values"] == [-16]
    assert box_plot_dict["high_values"] == [93]
    assert box_plot_dict["low_indices"] == [3]
    assert box_plot_dict["high_indices"] == [5]


def test_box_plot_on_non_numeric_col(outliers_df):
    error = "Cannot calculate box plot statistics for non-numeric column"

    non_numeric_series = init_series(
        outliers_df["non_numeric"],
        logical_type="Categorical",
    )
    with pytest.raises(TypeError, match=error):
        non_numeric_series.ww.box_plot_dict()

    wrong_dtype_series = init_series(
        outliers_df["has_outliers"],
        logical_type="Categorical",
    )
    with pytest.raises(TypeError, match=error):
        wrong_dtype_series.ww.box_plot_dict()


def test_box_plot_with_fully_null_col(outliers_df):
    fully_null_double_series = init_series(outliers_df["nans"], logical_type="Double")

    box_plot_dict = fully_null_double_series.ww.box_plot_dict()
    check_empty_box_plot_dict(box_plot_dict)

    box_plot_dict = fully_null_double_series.ww.box_plot_dict(
        quantiles={0.25: 1, 0.75: 10},
    )
    check_empty_box_plot_dict(box_plot_dict)

    fully_null_int_series = init_series(
        outliers_df["nans"],
        logical_type="IntegerNullable",
    )
    box_plot_dict = fully_null_int_series.ww.box_plot_dict()
    check_empty_box_plot_dict(box_plot_dict)

    fully_null_categorical_series = init_series(
        outliers_df["nans"],
        logical_type="Categorical",
    )
    error = "Cannot calculate box plot statistics for non-numeric column"
    with pytest.raises(TypeError, match=error):
        fully_null_categorical_series.ww.box_plot_dict()


def test_box_plot_with_empty_col(outliers_df):
    series = outliers_df["nans"].dropna()

    fully_null_double_series = init_series(series, logical_type="Double")

    box_plot_dict = fully_null_double_series.ww.box_plot_dict()
    check_empty_box_plot_dict(box_plot_dict)

    box_plot_dict = fully_null_double_series.ww.box_plot_dict(
        quantiles={0.25: 1, 0.75: 10},
    )
    check_empty_box_plot_dict(box_plot_dict)

    fully_null_int_series = init_series(series, logical_type="IntegerNullable")
    box_plot_dict = fully_null_int_series.ww.box_plot_dict()
    check_empty_box_plot_dict(box_plot_dict)

    fully_null_categorical_series = init_series(series, logical_type="Categorical")
    error = "Cannot calculate box plot statistics for non-numeric column"
    with pytest.raises(TypeError, match=error):
        fully_null_categorical_series.ww.box_plot_dict()


def test_box_plot_different_quantiles(outliers_df):
    has_outliers_series = outliers_df["has_outliers"]
    has_outliers_series.ww.init()

    # No quantiles passed in - they all get calculated
    box_plot_info = has_outliers_series.ww.box_plot_dict()

    assert set(box_plot_info.keys()) == {
        "method",
        "low_bound",
        "high_bound",
        "quantiles",
        "low_values",
        "high_values",
        "low_indices",
        "high_indices",
    }
    assert box_plot_info["method"] == "box_plot"
    assert box_plot_info["low_bound"] == 8.125
    assert box_plot_info["high_bound"] == 83.125
    assert len(box_plot_info["quantiles"]) == 5
    assert len(box_plot_info["high_values"]) == 1
    assert len(box_plot_info["low_values"]) == 1

    # The minimum required quantiles passed in with outliers present
    partial_quantiles = {0.0: -16, 0.25: 36.25, 0.75: 55.0, 1.0: 93}
    box_plot_info = has_outliers_series.ww.box_plot_dict(quantiles=partial_quantiles)

    assert set(box_plot_info.keys()) == {
        "method",
        "low_bound",
        "high_bound",
        "quantiles",
        "low_values",
        "high_values",
        "low_indices",
        "high_indices",
    }
    assert box_plot_info["low_bound"] == 8.125
    assert box_plot_info["high_bound"] == 83.125
    assert len(box_plot_info["quantiles"]) == 4
    assert len(box_plot_info["high_values"]) == 1
    assert len(box_plot_info["low_values"]) == 1

    # The minimum required quantiles passed in without outliers present
    no_outliers_series = outliers_df["no_outliers"]
    no_outliers_series.ww.init()
    partial_quantiles = {0.0: 23.0, 0.25: 36.25, 0.75: 55.0, 1.0: 60.0}
    box_plot_info = no_outliers_series.ww.box_plot_dict(quantiles=partial_quantiles)

    assert set(box_plot_info.keys()) == {
        "method",
        "low_bound",
        "high_bound",
        "quantiles",
        "low_values",
        "high_values",
        "low_indices",
        "high_indices",
    }
    assert box_plot_info["low_bound"] == 23.0
    assert box_plot_info["high_bound"] == 60.0
    assert len(box_plot_info["quantiles"]) == 4
    assert len(box_plot_info["high_values"]) == 0
    assert len(box_plot_info["low_values"]) == 0


def test_box_plot_quantiles_errors(outliers_df):
    series = outliers_df["has_outliers"]
    series.ww.init()

    error = re.escape(
        "Input quantiles do not contain the minimum necessary quantiles for box plot calculation: "
        "0.0 (the minimum value), 0.25 (the first quartile), 0.75 (the third quartile), and 1.0 (the maximum value).",
    )

    partial_quantiles = {0.25: 36.25, 0.75: 20, 1.0: 90}
    with pytest.raises(ValueError, match=error):
        series.ww.box_plot_dict(quantiles=partial_quantiles)

    empty_quantiles = {}
    with pytest.raises(ValueError, match=error):
        series.ww.box_plot_dict(quantiles=empty_quantiles)

    error = "quantiles must be a dictionary."
    with pytest.raises(TypeError, match=error):
        series.ww.box_plot_dict(quantiles=1)


def test_box_plot_optional_return_values(outliers_df):
    has_outliers_series = outliers_df["has_outliers"]
    has_outliers_series.ww.init()

    has_outliers_box_plot_info_without_optional = has_outliers_series.ww.box_plot_dict(
        include_indices_and_values=False,
    )
    has_outliers_box_plot_info_with_optional = has_outliers_series.ww.box_plot_dict(
        include_indices_and_values=True,
    )

    assert {"low_bound", "high_bound", "quantiles", "method"} == set(
        has_outliers_box_plot_info_without_optional.keys(),
    )
    assert {
        "method",
        "low_bound",
        "high_bound",
        "quantiles",
        "low_values",
        "high_values",
        "low_indices",
        "high_indices",
    } == set(has_outliers_box_plot_info_with_optional.keys())

    no_outliers_series = outliers_df["no_outliers"]
    no_outliers_series.ww.init()

    no_outliers_box_plot_info_without_optional = no_outliers_series.ww.box_plot_dict(
        include_indices_and_values=False,
    )
    no_outliers_box_plot_info_with_optional = no_outliers_series.ww.box_plot_dict(
        include_indices_and_values=True,
    )

    assert {"low_bound", "high_bound", "quantiles", "method"} == set(
        no_outliers_box_plot_info_without_optional.keys(),
    )
    assert {
        "method",
        "low_bound",
        "high_bound",
        "quantiles",
        "low_values",
        "high_values",
        "low_indices",
        "high_indices",
    } == set(no_outliers_box_plot_info_with_optional.keys())


def test_medcouple_outliers(skewed_outliers_df):
    outliers_series_skewed_right = skewed_outliers_df["right_skewed_outliers"]
    outliers_series_skewed_right.ww.init()

    outliers_series_skewed_left = skewed_outliers_df["left_skewed_outliers"]
    outliers_series_skewed_left.ww.init()

    right_skewed_dict = outliers_series_skewed_right.ww.medcouple_dict()
    left_skewed_dict = outliers_series_skewed_left.ww.medcouple_dict()

    assert set(right_skewed_dict.keys()) == {
        "method",
        "low_bound",
        "high_bound",
        "quantiles",
        "low_values",
        "high_values",
        "low_indices",
        "high_indices",
        "medcouple_stat",
    }

    expected_right_skewed_dict = {
        "method": "medcouple",
        "low_bound": 1.58676,
        "high_bound": 20.32873,
        "quantiles": {
            0.0: 1.0,
            0.25: 3.0,
            0.5: 4.0,
            0.75: 6.0,
            1.0: 30.0,
        },
        "low_values": [1, 1],
        "high_values": [30],
        "low_indices": [0, 1],
        "high_indices": [65],
        "medcouple_stat": 0.333,
    }

    expected_left_skewed_dict = {
        "method": "medcouple",
        "low_bound": 23.58676,
        "high_bound": 30.0,
        "quantiles": {
            0.0: 1.0,
            0.25: 25.0,
            0.5: 27.0,
            0.75: 28.0,
            1.0: 30.0,
        },
        "low_values": [23, 23, 22, 22, 21, 20, 18, 17, 15, 1],
        "high_values": [],
        "low_indices": [56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
        "high_indices": [],
        "medcouple_stat": -0.333,
    }

    assert right_skewed_dict == expected_right_skewed_dict
    assert left_skewed_dict == expected_left_skewed_dict

    outliers_series_skewed_right = skewed_outliers_df[
        "right_skewed_outliers_nullable_int"
    ]
    outliers_series_skewed_right.ww.init(logical_type="IntegerNullable")

    right_skewed_dict = outliers_series_skewed_right.ww.medcouple_dict()

    assert right_skewed_dict == expected_right_skewed_dict


def test_medcouple_outliers_with_quantiles(skewed_outliers_df):
    outliers_series_skewed_right = skewed_outliers_df["right_skewed_outliers"]
    outliers_series_skewed_right.ww.init()

    outliers_series_skewed_left = skewed_outliers_df["left_skewed_outliers"]
    outliers_series_skewed_left.ww.init()

    override_quantiles_right = {
        0.0: 1.0,
        0.25: 8.0,
        0.5: 11.0,
        0.75: 25.0,
        1.0: 30.0,
    }

    override_quantiles_left = {
        0.0: 1.0,
        0.25: 3.0,
        0.5: 4.0,
        0.75: 26.0,
        1.0: 30.0,
    }

    expected_skewed_dict = {
        "method": "medcouple",
        "low_bound": 1.0,
        "high_bound": 30.0,
        "quantiles": None,
        "low_values": [],
        "high_values": [],
        "low_indices": [],
        "high_indices": [],
        "medcouple_stat": 0.333,
    }

    right_skewed_dict = outliers_series_skewed_right.ww.medcouple_dict(
        quantiles=override_quantiles_right,
    )
    left_skewed_dict = outliers_series_skewed_left.ww.medcouple_dict(
        quantiles=override_quantiles_left,
    )

    expected_skewed_dict["quantiles"] = override_quantiles_right
    assert right_skewed_dict == expected_skewed_dict
    expected_skewed_dict["quantiles"] = override_quantiles_left
    expected_skewed_dict["medcouple_stat"] = -0.333
    assert left_skewed_dict == expected_skewed_dict


def test_get_outliers_for_column_with_nans_medcouple(skewed_outliers_df):
    contains_nans_series = skewed_outliers_df["has_outliers_with_nans"]
    contains_nans_series.ww.init()

    medcouple_dict = contains_nans_series.ww.medcouple_dict()

    expected_skewed_dict = {
        "method": "medcouple",
        "low_bound": 1.94779,
        "high_bound": 16.0754,
        "quantiles": {
            0.0: 1.0,
            0.25: 3.0,
            0.5: 4.0,
            0.75: 5.25,
            1.0: 30.0,
        },
        "low_values": [1.0, 1.0],
        "high_values": [30.0],
        "low_indices": [0, 1],
        "high_indices": [65],
        "medcouple_stat": 0.333,
    }

    assert medcouple_dict == expected_skewed_dict


@pytest.mark.parametrize("mc", [-1.0, -0.5, -0.1, 0, 0.3333333, 1.0])
def test_determine_coefficients(mc, skewed_outliers_df):
    if _is_spark_dataframe(skewed_outliers_df):
        pytest.xfail("spark hasn't implemented __iter__() for series")
    right_skewed = skewed_outliers_df["right_skewed_outliers"]
    left_skewed = skewed_outliers_df["left_skewed_outliers"]

    right_coeff = np.abs(skew(right_skewed))
    right_coeff = min(right_coeff, 3.5)

    left_coeff = np.abs(skew(left_skewed))
    left_coeff = min(left_coeff, 3.5)

    if mc >= 0:
        assert _determine_coefficients(right_skewed, mc) == (-right_coeff, right_coeff)
        assert _determine_coefficients(left_skewed, mc) == (-left_coeff, left_coeff)
    else:
        assert _determine_coefficients(right_skewed, mc) == (right_coeff, -right_coeff)
        assert _determine_coefficients(left_skewed, mc) == (left_coeff, -left_coeff)


def test_get_low_high_bound_warnings():
    error = "If the method selected is medcouple, then mc cannot be None."
    with pytest.raises(ValueError, match=error):
        _get_low_high_bound(None, "medcouple", 1, 1, None, None, None)

    error = "Acceptable methods are 'box_plot' and 'medcouple'. The value passed was 'something_else'"
    with pytest.raises(ValueError, match=error):
        _get_low_high_bound(None, "something_else", 1, 1, None, None, None)


def test_get_medcouple(outliers_df_pandas, skewed_outliers_df_pandas):
    has_outliers_series = outliers_df_pandas["has_outliers"]
    has_outliers_series = pd.concat(
        [has_outliers_series, pd.Series([39], dtype="int64")],
        ignore_index=True,
    )
    has_outliers_series.ww.init()
    mc = _get_medcouple_statistic(has_outliers_series)
    assert mc == 0.122

    outliers_series_skewed_right = skewed_outliers_df_pandas["right_skewed_outliers"]
    outliers_series_skewed_right.ww.init()
    mc = _get_medcouple_statistic(outliers_series_skewed_right)
    assert mc == 0.333

    outliers_series_skewed = skewed_outliers_df_pandas[
        ["right_skewed_outliers", "left_skewed_outliers"]
    ]
    outliers_series_skewed.ww.init()
    mc = _get_medcouple_statistic(outliers_series_skewed)
    assert isinstance(mc, np.ndarray)
    np.testing.assert_almost_equal(mc, np.array([0.33333333, -0.33333333]))


def test_determine_best_outlier_method_sampling_outcome(skewed_outliers_df_pandas):
    # Column of 66,000, far above the 10,000 limit
    contains_nans_series_skewed = (
        skewed_outliers_df_pandas["right_skewed_outliers"]
        .repeat(1000)
        .reset_index(drop=True)
    )
    contains_nans_series_skewed.ww.init()

    mc_result = _determine_best_outlier_method(contains_nans_series_skewed)

    assert mc_result.method == "medcouple"
    assert math.isclose(mc_result.mc, 0.33, rel_tol=0.01)


def test_determine_best_outlier_method_equivalent_outcome(
    outliers_df_pandas,
    skewed_outliers_df_pandas,
):
    contains_nans_series_skewed = skewed_outliers_df_pandas["right_skewed_outliers"]
    contains_nans_series_skewed.ww.init()

    contains_nans_series = outliers_df_pandas["has_outliers"]
    contains_nans_series.ww.init()

    outliers_mc_skewed = contains_nans_series_skewed.ww.get_outliers(method="medcouple")
    outliers_best_skewed = contains_nans_series_skewed.ww.get_outliers(method="best")

    outliers_bp = contains_nans_series.ww.get_outliers(method="box_plot")
    outliers_best = contains_nans_series.ww.get_outliers(method="best")

    assert "medcouple_stat" not in outliers_bp.keys()
    assert "medcouple_stat" in outliers_mc_skewed.keys()

    assert outliers_bp == outliers_best
    assert _get_medcouple_statistic(contains_nans_series) < 0.3

    assert outliers_mc_skewed == outliers_best_skewed
    assert _get_medcouple_statistic(contains_nans_series_skewed) >= 0.3


@patch.object(
    sys.modules["woodwork.statistics_utils._infer_temporal_frequencies"],
    "infer_frequency",
)
@pytest.mark.parametrize(
    "expected_call_args",
    [{}, {"debug": True}, {"temporal_columns": ["2D_freq", "3M_freq"]}],
)
def test_infer_temporal_frequencies(
    infer_frequency,
    expected_call_args,
    datetime_freqs_df_pandas,
):
    # TODO: Add support for Dask and Spark DataFrames
    datetime_freqs_df_pandas.ww.init()

    datetime_freqs_df_pandas.ww.infer_temporal_frequencies(**expected_call_args)

    expected_call_count = (
        len(expected_call_args["temporal_columns"])
        if "temporal_columns" in expected_call_args
        else 7
    )

    assert infer_frequency.called
    assert infer_frequency.call_count == expected_call_count

    actual_call_args = infer_frequency.call_args[1]

    window_length = config.get_option("frequence_inference_window_length")
    threshold = config.get_option("frequence_inference_threshold")

    expected_debug_flag = (
        expected_call_args["debug"] if "debug" in expected_call_args else False
    )

    assert actual_call_args["debug"] == expected_debug_flag
    assert actual_call_args["window_length"] == window_length
    assert actual_call_args["threshold"] == threshold


def test_infer_temporal_frequencies_with_columns(datetime_freqs_df_pandas):
    datetime_freqs_df_pandas.ww.init(time_index="2D_freq")

    frequency_dict = datetime_freqs_df_pandas.ww.infer_temporal_frequencies(
        temporal_columns=[datetime_freqs_df_pandas.ww.time_index],
    )
    assert len(frequency_dict) == 1
    assert frequency_dict["2D_freq"] == "2D"

    empty_frequency_dict = datetime_freqs_df_pandas.ww.infer_temporal_frequencies(
        temporal_columns=[],
    )
    assert len(empty_frequency_dict) == 0


def test_infer_temporal_frequencies_errors(datetime_freqs_df_pandas):
    datetime_freqs_df_pandas.ww.init()

    error = "Column not_present not found in dataframe."
    with pytest.raises(ValueError, match=error):
        datetime_freqs_df_pandas.ww.infer_temporal_frequencies(
            temporal_columns=["2D_freq", "not_present"],
        )

    error = "Cannot determine frequency for column ints with logical type Integer"
    with pytest.raises(TypeError, match=error):
        datetime_freqs_df_pandas.ww.infer_temporal_frequencies(
            temporal_columns=["1d_skipped_one_freq", "ints"],
        )


@pytest.mark.parametrize(
    ("measures", "expected"),
    [
        ("pearson", (["pearson"], ["pearson"], False)),
        (["pearson"], (["pearson"], ["pearson"], False)),
        ("spearman", (["spearman"], ["spearman"], False)),
        (["spearman"], (["spearman"], ["spearman"], False)),
        ("mutual_info", (["mutual_info"], ["mutual_info"], False)),
        (["mutual_info"], (["mutual_info"], ["mutual_info"], False)),
        ("max", (["max"], ["pearson", "spearman", "mutual_info"], True)),
        (["max"], (["max"], ["pearson", "spearman", "mutual_info"], True)),
        (
            "all",
            (
                ["max", "pearson", "spearman", "mutual_info"],
                ["pearson", "spearman", "mutual_info"],
                True,
            ),
        ),
        (
            ["all"],
            (
                ["max", "pearson", "spearman", "mutual_info"],
                ["pearson", "spearman", "mutual_info"],
                True,
            ),
        ),
        (
            ["mutual_info", "pearson"],
            (["mutual_info", "pearson"], ["pearson", "mutual_info"], False),
        ),
        (
            ["pearson", "max"],
            (["pearson", "max"], ["pearson", "spearman", "mutual_info"], True),
        ),
        (
            ["max", "pearson", "mutual_info"],
            (
                ["max", "pearson", "mutual_info"],
                ["pearson", "spearman", "mutual_info"],
                True,
            ),
        ),
        (
            ["pearson", "spearman"],
            (["pearson", "spearman"], ["pearson", "spearman"], False),
        ),
        (
            ["spearman", "max"],
            (["spearman", "max"], ["pearson", "spearman", "mutual_info"], True),
        ),
    ],
)
def test_parse_measures_valid(measures, expected):
    _measures, _calc_order, _calc_max = _parse_measures(measures)
    assert _measures == expected[0]
    assert _calc_order == expected[1]
    assert _calc_max == expected[2]


def test_parse_measures_warns():
    warning = "additional measures to 'all' measure found; 'all' should be used alone"
    with pytest.warns(ParametersIgnoredWarning, match=warning):
        _measures, _calc_order, _calc_max = _parse_measures(["pearson", "all"])
    assert _measures == ["max", "pearson", "spearman", "mutual_info"]
    assert _calc_order == ["pearson", "spearman", "mutual_info"]
    assert _calc_max


def test_parse_measures_wrong_input_types():
    msg = "Supplied measure 2 is not a string"
    with pytest.raises(TypeError, match=msg):
        _parse_measures(2)


def test_parse_measures_empty():
    msg = "No measures supplied"
    with pytest.raises(ValueError, match=msg):
        _parse_measures([])


def test_parse_measures_bad_string():
    msg = "Unrecognized dependence measure ruler"
    with pytest.raises(ValueError, match=msg):
        _parse_measures(["mutual_info", "ruler"])


@pytest.mark.parametrize("use_ordinal", [True, False])
def test_spearman_ordinal(df_mi, use_ordinal):
    if use_ordinal:
        df_mi.ww.init(logical_types={"strs2": Ordinal(order=["hi", "bye"])})
    else:
        df_mi.ww.init()
    sp = df_mi.ww.dependence(measures=["spearman"])
    valid_sp_columns = concat_dataframe_or_series(sp.column_1, sp.column_2).unique()
    assert "strs" not in valid_sp_columns
    if use_ordinal:
        assert "strs2" in valid_sp_columns
        return
    assert "strs2" not in valid_sp_columns


def test_dependence_target_col_not_exist(df_mi):
    df_mi.ww.init()
    with pytest.raises(ValueError, match="target_col 'value' not in the"):
        df_mi.ww.dependence_dict(target_col="value")

    with pytest.raises(ValueError, match="target_col 'value' not in the"):
        df_mi.ww.dependence(target_col="value")


def test_dependence_target_col_in_output(df_mi):
    df_mi.ww.init()
    dep = df_mi.ww.dependence_dict(min_shared=12, target_col="ints")
    assert all([x["column_2"] == "ints" for x in dep])
    assert len(dep) < len(df_mi.columns)
    assert all([isinstance(x["max"], float) for x in dep])


def test_dependence_dict_target_col_in_output(df_mi):
    df_mi.ww.init()
    dep = df_mi.ww.dependence(min_shared=12, target_col="ints")
    assert set(list(dep["column_2"])) == {"ints"}
    assert set(list(dep.columns) + ["all"]) == set(
        ["column_1", "column_2"] + CONFIG_DEFAULTS["correlation_metrics"],
    )


def test_convert_ordinal_to_numeric():
    df = pd.DataFrame(
        {
            "ints1": pd.Series([1, 2, 3, 2]),
            "ints2": pd.Series([1, 100, 1, 100]),
            "strs": pd.Series(["hi", "hi", "hi", "hi"]),
            "strs2": pd.Series(["bye", "hi", "bye", "bye"]),
            "bools": pd.Series([True, False, True, False]),
            "categories": pd.Series(["test", "test2", "test2", "test"]),
            "dates": pd.Series(
                ["2020-01-01", "2019-01-02", "2020-08-03", "1997-01-04"],
            ),
        },
    )
    df.ww.init(logical_types={"strs2": Ordinal(order=["hi", "bye"])})
    data = {column: df[column] for column in df.copy() if column != "ints2"}
    result = [0 if x == "hi" else 1 for x in data["strs2"].values]
    _convert_ordinal_to_numeric(df.ww.schema, data)
    for cols in data.keys():
        if cols != "strs2":
            assert all(data[cols] == df[cols])
        else:
            assert all(data["strs2"].values == result)


def test_dependence_with_object_target():
    df = pd.DataFrame(
        {
            "ints1": pd.Series([1, 2, 3, 2]),
            "strs": pd.Series(["hi", "hi", "hi", "hi"]),
            "target_y": pd.Series([True, False, False, pd.NA]),
        },
    )
    df.ww.init()
    res = df.ww.dependence(target_col="target_y")
    assert "pearson" in res.columns
    assert "spearman" in res.columns


def test_box_plot_ignore_zeros():
    zeros_df = pd.Series(list(range(1, 100)) + [0] * 100)
    no_zeros_df = pd.Series(range(1, 100))
    zeros_df.ww.init()
    no_zeros_df.ww.init()

    zeros_box_ignored = zeros_df.ww.box_plot_dict(ignore_zeros=True)
    zeros_box_not_ignored = zeros_df.ww.box_plot_dict()

    no_zeros_box_ignored = no_zeros_df.ww.box_plot_dict(ignore_zeros=True)
    no_zeros_box_not_ignored = no_zeros_df.ww.box_plot_dict()

    assert zeros_box_ignored == no_zeros_box_ignored
    assert zeros_box_ignored == no_zeros_box_not_ignored
    assert zeros_box_not_ignored != no_zeros_box_ignored


@pytest.mark.parametrize("dtype", ["IntegerNullable", "Double"])
def test_box_plot_ignore_zeros_null(dtype):
    zeros_df = pd.Series(list(range(1, 100)) + [0] * 100 + [None])
    no_zeros_df = pd.Series(list(range(1, 100)) + [None])
    zeros_df.ww.init(logical_type=dtype)
    no_zeros_df.ww.init(logical_type=dtype)

    zeros_box_ignored_skewed = zeros_df.ww.medcouple_dict(ignore_zeros=True)
    zeros_box_not_ignored_skewed = zeros_df.ww.medcouple_dict()

    zeros_box_ignored = zeros_df.ww.box_plot_dict(ignore_zeros=True)
    zeros_box_not_ignored = zeros_df.ww.box_plot_dict()

    no_zeros_box_ignored_skewed = no_zeros_df.ww.medcouple_dict(ignore_zeros=True)
    no_zeros_box_not_ignored_skewed = no_zeros_df.ww.medcouple_dict()

    no_zeros_box_ignored = no_zeros_df.ww.box_plot_dict(ignore_zeros=True)
    no_zeros_box_not_ignored = no_zeros_df.ww.box_plot_dict()

    assert zeros_box_ignored_skewed == no_zeros_box_ignored_skewed
    assert zeros_box_ignored_skewed == no_zeros_box_not_ignored_skewed
    assert zeros_box_not_ignored_skewed != no_zeros_box_ignored_skewed

    assert zeros_box_ignored == no_zeros_box_ignored
    assert zeros_box_ignored == no_zeros_box_not_ignored
    assert zeros_box_not_ignored != no_zeros_box_ignored
