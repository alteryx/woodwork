from get_ranges import get_ranges
import pandas as pd


def determine_duplicates_values(observed):
    observed_df = pd.DataFrame({"observed": observed}).reset_index(drop=True).diff()

    observed_dupes = observed_df[observed_df["observed"] == pd.Timedelta(0)]

    if len(observed_dupes) == 0:
        return []

    ranges = get_ranges(observed_dupes.index)
    out = []

    for start_idx, end_idx in ranges:
        out.append(
            RangeObject(
                observed[start_idx].isoformat(), start_idx, end_idx - start_idx + 1
            )
        )

    return out
