from ._get_ranges import _get_ranges
from ._constants import ESTIMATED_COLUMN_NAME, OBSERVED_COLUMN_NAME
from ._types import RangeObject
import pandas as pd


def _determine_nan_values(observed):
    """Calculate NaN Values in time_series:

    Args:
        observed_ts (Series): The observed time_series.

    Returns:
        (list(RangeObject)): A list of RangeObject data objects. A RangeObject has the following properties:

        - dt: an ISO 8610 formatted string of the first NaN timestamp
        - idx: first index of the NaN timestamp. Index is relative to estimated timeseries
        - range: the number of sequential elements that are NaN
    """

    observed_null = observed[observed.isnull()]

    if len(observed_null) == 0:
        return []

    ranges = _get_ranges(observed_null.index)
    out = []

    for start_idx, end_idx in ranges:
        out.append(
            RangeObject(
                dt = None,
                idx = start_idx, 
                range = end_idx - start_idx + 1
            )
        )

    return out
