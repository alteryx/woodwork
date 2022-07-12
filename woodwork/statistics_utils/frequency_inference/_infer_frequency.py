import dataclasses

import pandas as pd

from woodwork.statistics_utils.frequency_inference._clean_timeseries import (
    _clean_timeseries,
)
from woodwork.statistics_utils.frequency_inference._constants import (
    FREQ_INFERENCE_THRESHOLD,
    WINDOW_LENGTH,
)
from woodwork.statistics_utils.frequency_inference._determine_duplicate_values import (
    _determine_duplicate_values,
)
from woodwork.statistics_utils.frequency_inference._determine_extra_values import (
    _determine_extra_values,
)
from woodwork.statistics_utils.frequency_inference._determine_missing_values import (
    _determine_missing_values,
)
from woodwork.statistics_utils.frequency_inference._determine_most_likely_freq import (
    _determine_most_likely_freq,
)
from woodwork.statistics_utils.frequency_inference._determine_nan_values import (
    _determine_nan_values,
)
from woodwork.statistics_utils.frequency_inference._generate_estimated_timeseries import (
    _generate_estimated_timeseries,
)
from woodwork.statistics_utils.frequency_inference._generate_freq_candidates import (
    _generate_freq_candidates,
)
from woodwork.statistics_utils.frequency_inference._types import (
    DataCheckMessageCode,
    InferDebug,
)


def inference_response(inferred_freq, debug_obj, debug):
    if debug:
        return (inferred_freq, dataclasses.asdict(debug_obj))
    else:
        return inferred_freq


def infer_frequency(
    observed_ts: pd.Series,
    debug=False,
    window_length=WINDOW_LENGTH,
    threshold=FREQ_INFERENCE_THRESHOLD,
):
    """Infer the frequency of a given Pandas Datetime Series.

    Args:
        series (pd.Series): data to use for histogram
        debug (boolean): a flag to determine if debug object should be returned (explained below).
        window_length (int): the window length used to determine the most likely candidate frequence. Default is 15. If the timeseries is noisy
            and needs to inferred, the minimum length of the input timeseries needs to be greater than this window.
        threshold (float): a value between 0 and 1. Given the number of windows that contain the most observed frequency (N), and total number of windows (T),
            if N/T > threshold, the most observed frequency is determined to be the most likely frequency, else None.

    Returns:
        inferred_freq (str): pandas offset alias string (D, M, Y, etc.) or None if no uniform frequency was present in the data.
        debug (dict): a dictionary containing debug information if frequency cannot be inferred. This dictionary has the following properties:

        - actual_range_start (str): a string representing the minimum Timestamp in the input observed timeseries according to ISO 8601.
        - actual_range_end (str): a string representing the maximum Timestamp in the input observed timeseries according to ISO 8601.

        - message (str): message describing any issues with the input Datetime series

        - estimated_freq (str): None
        - estimated_range_start (str): a string representing the minimum Timestamp in the output estimated timeseries according to ISO 8601.
        - estimated_range_end (str): a string representing the maximum Timestamp in the output estimated timeseries according to ISO 8601.

        - duplicate_values (list(RangeObject)): a list of RangeObjects of Duplicate timestamps
        - missing_values (list(RangeObject)): a list of RangeObjects of Missing timestamps
        - extra_values (list(RangeObject)): a list of RangeObjects of Extra timestamps
        - nan_values (list(RangeObject)): a list of RangeObjects of NaN timestamps

        A range object contains the following information:

        - dt: an ISO 8601 formatted string of the first timestamp in this range
        - idx: the index of the first timestamp in this range
            - for duplicates and extra values, the idx is in reference to the observed data
            - for missing values, the idx is in reference to the estimated data.
        - range: the length of this range.
    """

    pandas_inferred_freq = pd.infer_freq(observed_ts)

    if pandas_inferred_freq or not debug:
        return inference_response(
            inferred_freq=pandas_inferred_freq,
            debug_obj=InferDebug(),
            debug=debug,
        )

    # clean observed timeseries from duplicates and NaTs
    observed_ts_clean = _clean_timeseries(observed_ts)

    # Determine if series is not empty
    if len(observed_ts_clean) == 0:
        return inference_response(
            inferred_freq=None,
            debug_obj=InferDebug(
                message=DataCheckMessageCode.DATETIME_SERIES_IS_EMPTY,
            ),
            debug=debug,
        )

    nan_values = _determine_nan_values(observed_ts)
    duplicate_values = _determine_duplicate_values(observed_ts)

    actual_range_start = observed_ts_clean.min().isoformat()
    actual_range_end = observed_ts_clean.max().isoformat()

    # Determine if series is long enough for inference
    if len(observed_ts_clean) < window_length:
        return inference_response(
            inferred_freq=None,
            debug_obj=InferDebug(
                actual_range_start=actual_range_start,
                actual_range_end=actual_range_end,
                message=DataCheckMessageCode.DATETIME_SERIES_IS_NOT_LONG_ENOUGH,
                duplicate_values=duplicate_values,
                nan_values=nan_values,
            ),
            debug=debug,
        )

    # Determine if series if Monotonic
    is_monotonic = observed_ts_clean.is_monotonic_increasing
    if not is_monotonic:
        return inference_response(
            inferred_freq=None,
            debug_obj=InferDebug(
                actual_range_start,
                actual_range_end,
                message=DataCheckMessageCode.DATETIME_SERIES_IS_NOT_MONOTONIC,
                duplicate_values=duplicate_values,
                nan_values=nan_values,
            ),
            debug=debug,
        )

    # Generate Frequency Candidates
    alias_dict = _generate_freq_candidates(
        observed_ts_clean,
        window_length=window_length,
    )

    most_likely_freq = _determine_most_likely_freq(alias_dict, threshold=threshold)

    if most_likely_freq is None:
        return inference_response(
            inferred_freq=None,
            debug_obj=InferDebug(
                actual_range_start,
                actual_range_end,
                DataCheckMessageCode.DATETIME_SERIES_FREQ_CANNOT_BE_ESTIMATED,
                duplicate_values=duplicate_values,
                nan_values=nan_values,
            ),
            debug=debug,
        )

    most_likely_freq_alias_dict = alias_dict[most_likely_freq]

    estimated_ts = _generate_estimated_timeseries(most_likely_freq_alias_dict)

    estimated_range_start = most_likely_freq_alias_dict["min_dt"].isoformat()
    estimated_range_end = most_likely_freq_alias_dict["max_dt"].isoformat()

    missing_values = _determine_missing_values(estimated_ts, observed_ts_clean)
    extra_values = _determine_extra_values(estimated_ts, observed_ts_clean)

    return inference_response(
        inferred_freq=None,
        debug_obj=InferDebug(
            actual_range_start=actual_range_start,
            actual_range_end=actual_range_end,
            estimated_freq=most_likely_freq,
            estimated_range_start=estimated_range_start,
            estimated_range_end=estimated_range_end,
            missing_values=missing_values,
            duplicate_values=duplicate_values,
            extra_values=extra_values,
            nan_values=nan_values,
        ),
        debug=debug,
    )
