from woodwork.statistics_utils.frequency_inference._get_ranges import _get_ranges
from woodwork.statistics_utils.frequency_inference._types import RangeObject


def _determine_nan_values(observed):
    """Calculate NaN Values in time_series:

    Args:
        observed_ts (Series): The observed time_series.

    Returns:
        (list(RangeObject)): A list of RangeObject data objects. A RangeObject has the following properties:

        - dt: an ISO 8601 formatted string of the first NaN timestamp
        - idx: first index of the NaN timestamp. Index is relative to estimated timeseries
        - range: the number of sequential elements that are NaN
    """

    observed_null = observed[observed.isnull()]

    if len(observed_null) == 0:
        return []

    ranges = _get_ranges(observed_null.index)

    return [
        RangeObject(dt=None, idx=start_idx, range=end_idx - start_idx + 1)
        for start_idx, end_idx in ranges
    ]
