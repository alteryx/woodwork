import pandas as pd

from woodwork.statistics_utils.frequency_inference._constants import (
    OBSERVED_COLUMN_NAME,
)
from woodwork.statistics_utils.frequency_inference._get_ranges import _get_ranges
from woodwork.statistics_utils.frequency_inference._types import RangeObject


def _determine_duplicate_values(observed_ts):
    """Calculate Duplicate Values in time_series:

    Args:
        observed_ts (Series): The observed time_series.

    Returns:
        (list(RangeObject)): A list of RangeObject data objects. A RangeObject has the following properties:

        - dt: an ISO 8601 formatted string of the duplicate timestamp
        - idx: first index of duplicate timestamp. Index is relative to observed timeseries
        - range: the number of sequential elements that are duplicated
    """

    observed_df = (
        pd.DataFrame({OBSERVED_COLUMN_NAME: observed_ts}).reset_index(drop=True).diff()
    )

    observed_dupes = observed_df[observed_df[OBSERVED_COLUMN_NAME] == pd.Timedelta(0)]

    if len(observed_dupes) == 0:
        return []

    ranges = _get_ranges(observed_dupes.index)

    return [
        RangeObject(
            dt=observed_ts[start_idx].isoformat(),
            idx=int(start_idx),
            range=int(end_idx - start_idx + 1),
        )
        for start_idx, end_idx in ranges
    ]
