import pandas as pd
from ._constants import NON_INFERABLE_FREQ, WINDOW_LENGTH, CANDIDATE_COLUMN_NAME


def _generate_freq_candidates(time_series):
    candidates = [[] for x in range(len(time_series))]
    window_idx = 0
    alias_dict = {}
    for window in time_series.rolling(WINDOW_LENGTH):
        if len(window) == WINDOW_LENGTH:

            # calculate alias 
            alias = pd.infer_freq(window) or NON_INFERABLE_FREQ

            if alias in alias_dict:
                alias_dict[alias] += 1
            else:
                alias_dict[alias] = 1

            for i in range(window_idx, window_idx + WINDOW_LENGTH):
                candidates[i].append(alias)
            window_idx += 1

    candidate_df = pd.DataFrame({CANDIDATE_COLUMN_NAME: candidates}, index=time_series)

    return (candidate_df, alias_dict)
