import pandas as pd

from woodwork.statistics_utils.frequency_inference._constants import (
    ESTIMATED_COLUMN_NAME,
    OBSERVED_COLUMN_NAME,
)
from woodwork.statistics_utils.frequency_inference._get_ranges import _get_ranges
from woodwork.statistics_utils.frequency_inference._types import RangeObject


def _determine_extra_values(estimated_ts, observed_ts):
    """Calculate Extra Values in time_series:

    Args:
        estimated_ts (Series): The estimated time_series.
        observed_ts (Series): The observed time_series.

    Returns:
        (list(RangeObject)): A list of RangeObject data objects. A RangeObject has the following properties:

        - dt: an ISO 8601 formatted string of the first extra timestamp
        - idx: first index of the extra timestamp. Index is relative to observed timeseries
        - range: the number of sequential elements that are extra
    """
    estimated_df = pd.DataFrame({ESTIMATED_COLUMN_NAME: estimated_ts})
    observed_df = pd.DataFrame({OBSERVED_COLUMN_NAME: observed_ts})

    merged_df = estimated_df.merge(
        observed_df,
        how="right",
        left_on=ESTIMATED_COLUMN_NAME,
        right_on=OBSERVED_COLUMN_NAME,
    )

    estimated_null = merged_df[merged_df[ESTIMATED_COLUMN_NAME].isnull()]

    if len(estimated_null) == 0:
        return []

    ranges = _get_ranges(estimated_null.index)

    return [
        RangeObject(
            dt=observed_ts.iloc[start_idx].isoformat(),
            idx=observed_ts.index[start_idx],
            range=end_idx - start_idx + 1,
        )
        for start_idx, end_idx in ranges
    ]
