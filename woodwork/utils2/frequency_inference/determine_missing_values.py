from .get_ranges import _get_ranges
from .constants import ESTIMATED_COLUMN_NAME, OBSERVED_COLUMN_NAME
import pandas as pd


def _determine_missing_values(estimated, observed):
    estimated_df = pd.DataFrame({ESTIMATED_COLUMN_NAME: estimated})
    observed_df = pd.DataFrame({ESTIMATED_COLUMN_NAME: observed})

    merged_df = estimated_df.merge(
        observed_df,
        how="left",
        left_on=ESTIMATED_COLUMN_NAME,
        right_on=ESTIMATED_COLUMN_NAME,
    )

    observed_null = merged_df[merged_df[ESTIMATED_COLUMN_NAME].isnull()]

    if len(observed_null) == 0:
        return []
    ranges = _get_ranges(observed_null.index)
    out = []

    for start_idx, end_idx in ranges:
        out.append(
            RangeObject(
                estimated[start_idx].isoformat(), start_idx, end_idx - start_idx + 1
            )
        )

    return out
