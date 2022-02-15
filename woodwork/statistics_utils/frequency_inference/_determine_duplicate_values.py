from ._get_ranges import _get_ranges
from ._constants import OBSERVED_COLUMN_NAME
from ._types import RangeObject
import pandas as pd


def _determine_duplicate_values(observed):
    observed_df = (
        pd.DataFrame({OBSERVED_COLUMN_NAME: observed}).reset_index(drop=True).diff()
    )

    observed_dupes = observed_df[observed_df[OBSERVED_COLUMN_NAME] == pd.Timedelta(0)]

    if len(observed_dupes) == 0:
        return []

    ranges = _get_ranges(observed_dupes.index)
    out = []

    for start_idx, end_idx in ranges:
        out.append(
            RangeObject(
                observed[start_idx].isoformat(), start_idx, end_idx - start_idx + 1
            )
        )

    return out
