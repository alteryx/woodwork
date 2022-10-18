import numpy as np
import pandas as pd
import pytest

from woodwork.statistics_utils import infer_frequency
from woodwork.tests.fixtures.datetime_freq import (
    generate_infer_error_messages,
    generate_pandas_inferrable,
    inferrable_freq_fixtures,
)


@pytest.mark.parametrize("freq", ["H", "2D", "3W-FRI", "M", "B", "Q-NOV"])
@pytest.mark.parametrize("error_range", [1, 2, 10])
def test_inferable_temporal_frequencies_missing(freq, error_range):
    # strip off first element, since it probably doesn't agree with freq
    dates = (
        (pd.date_range("2005-01-01", periods=1001, freq=freq)[1:])
        .to_series()
        .reset_index(drop=True)
    )

    actual_range_start = dates.loc[0].isoformat()
    actual_range_end = dates.loc[len(dates) - 1].isoformat()

    idx = len(dates) // 2

    d = dates[idx].isoformat()

    dates_observed = dates.drop(dates.loc[idx : idx + error_range - 1].index)
    dates_observed = dates_observed.reset_index(drop=True)

    expected_debug_obj = {
        "actual_range_start": actual_range_start,
        "actual_range_end": actual_range_end,
        "message": None,
        "estimated_freq": freq,
        "estimated_range_start": dates_observed.loc[0].isoformat(),
        "estimated_range_end": dates_observed.loc[len(dates_observed) - 1].isoformat(),
        "duplicate_values": [],
        "missing_values": [
            {"dt": d, "idx": idx, "range": error_range},
        ],
        "extra_values": [],
        "nan_values": [],
    }

    inferred_freq, actual_debug_obj = infer_frequency(
        observed_ts=dates_observed,
        debug=True,
    )

    assert inferred_freq is None
    assert actual_debug_obj == expected_debug_obj


@pytest.mark.parametrize("freq", ["H", "2D", "3W-FRI", "M", "B", "Q-NOV"])
@pytest.mark.parametrize("error_range", [1, 2, 10])
def test_inferable_temporal_frequencies_duplicates(freq, error_range):
    # strip off first element, since it probably doesn't agree with freq
    dates = (
        (pd.date_range("2005-01-01", periods=1001, freq=freq)[1:])
        .to_series()
        .reset_index(drop=True)
    )

    actual_range_start = dates.loc[0].isoformat()
    actual_range_end = dates.loc[len(dates) - 1].isoformat()

    idx = len(dates) // 2

    d = dates[idx].isoformat()

    dates_observed = pd.concat(
        [
            dates[: idx + 1],
            pd.Series(np.full((error_range,), dates[idx])),
            dates[idx + 1 :],
        ],
    )
    dates_observed = dates_observed.reset_index(drop=True)

    expected_debug_obj = {
        "actual_range_start": actual_range_start,
        "actual_range_end": actual_range_end,
        "message": None,
        "estimated_freq": freq,
        "estimated_range_start": dates_observed.loc[0].isoformat(),
        "estimated_range_end": dates_observed.loc[len(dates_observed) - 1].isoformat(),
        "duplicate_values": [
            {"dt": d, "idx": idx + 1, "range": error_range},
        ],
        "missing_values": [],
        "extra_values": [],
        "nan_values": [],
    }

    inferred_freq, actual_debug_obj = infer_frequency(
        observed_ts=dates_observed,
        debug=True,
    )

    assert inferred_freq is None
    assert actual_debug_obj == expected_debug_obj


@pytest.mark.parametrize("freq", ["H", "2D", "3W-FRI", "M", "B", "Q-NOV"])
@pytest.mark.parametrize("error_range", [1, 2, 10])
def test_inferable_temporal_frequencies_extra(freq, error_range):
    # strip off first element, since it probably doesn't agree with freq
    dates = (
        (pd.date_range("2005-01-01", periods=1001, freq=freq)[1:])
        .to_series()
        .reset_index(drop=True)
    )

    actual_range_start = dates.loc[0].isoformat()
    actual_range_end = dates.loc[len(dates) - 1].isoformat()

    idx = len(dates) // 2

    extra = (
        pd.date_range(dates[idx], periods=error_range + 1, freq="N")[1:]
    ).to_series()
    d = extra[0].isoformat()
    dates_observed = pd.concat([dates[: idx + 1], extra, dates[idx + 1 :]])
    dates_observed = dates_observed.reset_index(drop=True)

    expected_debug_obj = {
        "actual_range_start": actual_range_start,
        "actual_range_end": actual_range_end,
        "message": None,
        "estimated_freq": freq,
        "estimated_range_start": dates_observed.loc[0].isoformat(),
        "estimated_range_end": dates_observed.loc[len(dates_observed) - 1].isoformat(),
        "duplicate_values": [],
        "missing_values": [],
        "extra_values": [
            {"dt": d, "idx": idx + 1, "range": error_range},
        ],
        "nan_values": [],
    }

    inferred_freq, actual_debug_obj = infer_frequency(
        observed_ts=dates_observed,
        debug=True,
    )

    assert inferred_freq is None
    assert actual_debug_obj == expected_debug_obj


@pytest.mark.parametrize("case", inferrable_freq_fixtures)
def test_inferable_temporal_frequencies_cases(case):
    input_series = case["dates"]

    expected_debug_obj = case["expected_debug_obj"]

    inferred_freq, actual_debug_obj = infer_frequency(
        observed_ts=input_series,
        debug=True,
    )

    assert inferred_freq is None
    assert actual_debug_obj == expected_debug_obj


@pytest.mark.parametrize("case", generate_pandas_inferrable())
def test_pandas_inferable_temporal_frequencies(case):
    input_series = case["dates"]

    expected_infer_freq = case["expected_infer_freq"]

    inferred_freq = infer_frequency(observed_ts=input_series, debug=False)

    assert inferred_freq == expected_infer_freq


@pytest.mark.parametrize("case", generate_infer_error_messages())
def test_error_messages(case):
    input_series = case["dates"]

    expected_debug_obj = case["expected_debug_obj"]

    inferred_freq, actual_debug_obj = infer_frequency(
        observed_ts=input_series,
        debug=True,
    )

    assert inferred_freq is None
    assert actual_debug_obj == expected_debug_obj
