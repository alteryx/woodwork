import pandas as pd
from ._constants import NON_INFERABLE_FREQ, WINDOW_LENGTH, CANDIDATE_COLUMN_NAME


def _generate_freq_candidates(time_series):
    """Calculate a set of candidate frequecies for incoming time_series

    Args:
        time_series (Series): The time_series for which candidate frequencies will be calculated.

    Returns:
        (dict[pandas_alias(str) -> dict]): Returns a dictionary where each key is candidate Pandas Frequency Alias and the value is
        a dictionary containing the following keys:
            - alias (str): The Pandas Freq Alias
            - count (int): The number of windows where this alias is valid
            - min_dt (pd.TimeStamp): The earliest timestamp for this frequency.
            - max_dt (pd.TimeStamp): The latest timestamp for this frequency.
    """
    
    alias_dict = {}
    for window in time_series.rolling(WINDOW_LENGTH):
        if len(window) == WINDOW_LENGTH:

            # calculate alias 
            alias = pd.infer_freq(window) or NON_INFERABLE_FREQ

            min_dt = window.min()
            max_dt = window.max()

            if alias in alias_dict:
                curr_alias = alias_dict[alias]
                curr_alias["count"] += 1
                curr_alias["max_dt"] = window.iloc[WINDOW_LENGTH-1]

            else:
                alias_dict[alias] = {
                    "alias": alias,
                    "min_dt": min_dt,
                    "max_dt": max_dt,
                    "count": 1
                }

    return alias_dict
