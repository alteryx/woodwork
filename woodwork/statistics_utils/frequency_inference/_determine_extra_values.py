from tracemalloc import start
from ._get_ranges import _get_ranges
from ._constants import ESTIMATED_COLUMN_NAME, OBSERVED_COLUMN_NAME
from ._types import RangeObject
import pandas as pd


def _determine_extra_values(estimated, observed):
    """Calculate Extra Values in time_series:

    Args:
        observed_ts (Series): The observed time_series.

    Returns:
        (list(RangeObject)): A list of RangeObject data objects. A RangeObject has the following properties:

        - dt: an ISO 8610 formatted string of the first extra timestamp
        - idx: first index of the extra timestamp. Index is relative to observed timeseries
        - range: the number of sequential elements that are extra
    """
    estimated_df = pd.DataFrame({ESTIMATED_COLUMN_NAME: estimated})
    observed_df = pd.DataFrame({OBSERVED_COLUMN_NAME: observed})

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
    out = []

    for start_idx, end_idx in ranges:
        out.append(
            RangeObject(
                observed.iloc[start_idx].isoformat(), observed.index[start_idx], end_idx - start_idx + 1
            )
        )

    return out
