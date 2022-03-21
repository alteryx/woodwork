import pandas as pd


def _clean_timeseries(observed_ts: pd.Series) -> pd.Series:
    """Clean and Normalize time_series for subsequent processing. The following is performed on the time_series:

    - index_reset
    - duplicates dropped
    - na values dropped

    Args:
        observed_ts (Series): The time_series to normalize.

    Returns:
        (Series): The normalized time_series
    """

    observed_ts = observed_ts.reset_index(drop=True)
    observed_ts = observed_ts.drop_duplicates()
    observed_ts = observed_ts.dropna()
    return observed_ts
