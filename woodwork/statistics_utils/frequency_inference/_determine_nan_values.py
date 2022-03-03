from ._get_ranges import _get_ranges
from ._constants import ESTIMATED_COLUMN_NAME, OBSERVED_COLUMN_NAME
from ._types import RangeObject
import pandas as pd


def _determine_nan_values(observed):
   
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
