from ._get_ranges import _get_ranges
from ._constants import ESTIMATED_COLUMN_NAME, OBSERVED_COLUMN_NAME
from ._types import RangeObject
import pandas as pd


def _determine_extra_values(estimated, observed):
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
                observed[start_idx].isoformat(), start_idx, end_idx - start_idx + 1
            )
        )

    return out
