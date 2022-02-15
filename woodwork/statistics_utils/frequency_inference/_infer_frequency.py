import dataclasses
from ._generate_freq_candidates import _generate_freq_candidates
from ._determine_most_likely_freq import _determine_most_likely_freq
from ._build_freq_dataframe import _build_freq_dataframe
from ._generate_estimated_timeseries import _generate_estimated_timeseries
from ._determine_missing_values import _determine_missing_values
from ._determine_duplicate_values import _determine_duplicate_values
from ._determine_extra_values import _determine_extra_values
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

    actual_range_start = observed_ts.min().isoformat()
    actual_range_end = observed_ts.max().isoformat()

    # Determine if series if Monotonic
    is_monotonic = observed_ts.is_monotonic_increasing
    if not is_monotonic:
        return (
            None,
            InferDebug(
                actual_range_start,
                actual_range_end,
                DataCheckMessageCode.DATETIME_IS_NOT_MONOTONIC,
            ),
        )

    # Generate Frequency Candidates

    observed_ts_no_dupes = observed_ts.drop_duplicates()
    candidate_df, alias_dict = _generate_freq_candidates(observed_ts_no_dupes)

    most_likely_freq = _determine_most_likely_freq(alias_dict)

    if most_likely_freq is None:
        return (
            None,
            InferDebug(
                actual_range_start,
                actual_range_end,
                DataCheckMessageCode.DATETIME_FREQ_CANNOT_BE_ESTIMATED,
            ),
        )

    # Build Freq Dataframe, get alias_dict
    freq_df = _build_freq_dataframe(candidate_df)

    estimated_ts = _generate_estimated_timeseries(freq_df, most_likely_freq)

    estimated_range_start = estimated_ts.min().isoformat()
    estimated_range_end = estimated_ts.max().isoformat()

    missing = _determine_missing_values(estimated_ts, observed_ts)
    extra = _determine_extra_values(estimated_ts, observed_ts)
    duplicates = _determine_duplicate_values(observed_ts)

    return dataclasses.asdict(
        InferDebug(
            actual_range_start=actual_range_start,
            actual_range_end=actual_range_end,
            estimated_freq=most_likely_freq,
            estimated_range_start=estimated_range_start,
            estimated_range_end=estimated_range_end,
            missing_values=missing,
            duplicate_values=duplicates,
            extra_values=extra,
        )
    )
