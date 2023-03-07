import pandas as pd

from woodwork.statistics_utils.frequency_inference._constants import (
    NON_INFERABLE_FREQ,
    WINDOW_LENGTH,
)


def _generate_freq_candidates(time_series, window_length=WINDOW_LENGTH):
    """Calculate a set of candidate frequencies for incoming time_series

    Args:
        time_series (Series): The time_series for which candidate frequencies will be calculated.
        window_length (int): The length of the window. Default is 15

    Returns:
        (dict[pandas_alias(str) -> dict]): Returns a dictionary where each key is candidate Pandas Frequency Alias and the value is
        a dictionary containing the following keys:
            - alias (str): The Pandas Freq Alias
            - count (int): The number of windows where this alias is valid
            - min_dt (pd.TimeStamp): The earliest timestamp for this frequency.
            - max_dt (pd.TimeStamp): The latest timestamp for this frequency.
    """

    alias_dict = {}
    for window in time_series.rolling(window_length):
        if len(window) == window_length:
            # calculate alias
            alias = pd.infer_freq(window) or NON_INFERABLE_FREQ

            min_dt = window.min()
            max_dt = window.max()

            if alias in alias_dict:
                curr_alias = alias_dict[alias]
                curr_alias["count"] += 1
                curr_alias["max_dt"] = window.iloc[window_length - 1]

            else:
                alias_dict[alias] = {
                    "alias": alias,
                    "min_dt": min_dt,
                    "max_dt": max_dt,
                    "count": 1,
                }

    return alias_dict
