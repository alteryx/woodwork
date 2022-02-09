from get_ranges import get_ranges
import pandas as pd


def determine_missing_values(estimated, observed):
    estimated_df = pd.DataFrame({"estimated": estimated})
    observed_df = pd.DataFrame({"observed": observed})

    merged_df = estimated_df.merge(
        observed_df, how="left", left_on="estimated", right_on="observed"
    )

    observed_null = merged_df[merged_df["observed"].isnull()]

    if len(observed_null) == 0:
        return []
    ranges = get_ranges(observed_null.index)
    out = []

    for start_idx, end_idx in ranges:
        out.append(
            RangeObject(
                estimated[start_idx].isoformat(), start_idx, end_idx - start_idx + 1
            )
        )

    return out
