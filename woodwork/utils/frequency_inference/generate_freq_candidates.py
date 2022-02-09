import pandas as pd
from woodwork.utils.frequency_inference.constants import (
    NON_INFERABLE_FREQ,
    WINDOW_LENGTH,
)


def generate_freq_candidates(time_series):
    candidates = [[] for x in range(len(time_series))]
    window_idx = 0
    alias_dict = {}
    for window in time_series.rolling(WINDOW_LENGTH):
        if len(window) == WINDOW_LENGTH:
            alias = pd.infer_freq(window) or NON_INFERABLE_FREQ

            if alias in alias_dict:
                alias_dict[alias] += 1
            else:
                alias_dict[alias] = 1

            for i in range(window_idx, window_idx + WINDOW_LENGTH):
                try:
                    candidates[i].append(alias)
                except IndexError:
                    candidates[i] = [alias]
            window_idx += 1

    candidate_df = pd.DataFrame({"candidates": candidates}, index=time_series)

    return (candidate_df, alias_dict)
