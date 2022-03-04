import dataclasses
from ._generate_freq_candidates import _generate_freq_candidates
from ._determine_most_likely_freq import _determine_most_likely_freq
from ._build_freq_dataframe import _build_freq_dataframe
from ._generate_estimated_timeseries import _generate_estimated_timeseries
from ._determine_missing_values import _determine_missing_values
from ._determine_duplicate_values import _determine_duplicate_values
from ._determine_nan_values import _determine_nan_values
from ._determine_extra_values import _determine_extra_values
from ._clean_timeseries import _clean_timeseries
from ._types import InferDebug, DataCheckMessageCode

import pandas as pd


def infer_frequency(observed_ts: pd.Series):
    """Infer the frequency of a given Pandas Datetime Series.

    Args:
        series (pd.Series): data to use for histogram

    Returns:
        inferred_freq: a string
        histogram (list(dict)): a list of dictionary with keys `bins` and
            `frequency`
    """
    # clean observed timeseries from duplicates and NaTs
    observed_ts_clean = _clean_timeseries(observed_ts)

    # Determine if series is not empty
    if len(observed_ts_clean) == 0:
        return (
            None,
            InferDebug(
                message = DataCheckMessageCode.DATETIME_SERIES_IS_EMPTY,
            ),
        )

    nan_values = _determine_nan_values(observed_ts)
    duplicate_values = _determine_duplicate_values(observed_ts)

    actual_range_start = observed_ts_clean.min().isoformat()
    actual_range_end = observed_ts_clean.max().isoformat()

    # Determine if series is long enough for inference
    if len(observed_ts_clean) < 3:
        return (
            None,
            InferDebug(
                actual_range_start=actual_range_start,
                actual_range_end=actual_range_end,
                message = DataCheckMessageCode.DATETIME_SERIES_IS_NOT_LONG_ENOUGH,
                duplicate_values=duplicate_values,
                nan_values=nan_values
            ),
        )


    # Determine if series if Monotonic
    is_monotonic = observed_ts_clean.is_monotonic_increasing
    if not is_monotonic:
        return (
            None,
            InferDebug(
                actual_range_start,
                actual_range_end,
                message = DataCheckMessageCode.DATETIME_IS_NOT_MONOTONIC,
                duplicate_values=duplicate_values,
                nan_values=nan_values
            ),
        )

    # Generate Frequency Candidates
    candidate_df, alias_dict = _generate_freq_candidates(observed_ts_clean)

    most_likely_freq = _determine_most_likely_freq(alias_dict)

    if most_likely_freq is None:
        return (
            None,
            InferDebug(
                actual_range_start,
                actual_range_end,
                DataCheckMessageCode.DATETIME_FREQ_CANNOT_BE_ESTIMATED,
                duplicate_values=duplicate_values,
                nan_values=nan_values
            ),
        )

    # Build Freq Dataframe, get alias_dict
    freq_df = _build_freq_dataframe(candidate_df)

    estimated_ts = _generate_estimated_timeseries(freq_df, most_likely_freq)

    estimated_range_start = estimated_ts.min().isoformat()
    estimated_range_end = estimated_ts.max().isoformat()

    missing_values = _determine_missing_values(estimated_ts, observed_ts_clean)
    extra_values = _determine_extra_values(estimated_ts, observed_ts_clean)

    return dataclasses.asdict(
        InferDebug(
            actual_range_start=actual_range_start,
            actual_range_end=actual_range_end,
            estimated_freq=most_likely_freq,
            estimated_range_start=estimated_range_start,
            estimated_range_end=estimated_range_end,
            missing_values=missing_values,
            duplicate_values=duplicate_values,
            extra_values=extra_values,
            nan_values=nan_values
        )
    )
