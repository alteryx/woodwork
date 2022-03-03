import pandas as pd

def _clean_timeseries(observed_ts: pd.Series) -> pd.Series:
    observed_ts = observed_ts.copy()
    observed_ts = observed_ts.drop_duplicates()
    observed_ts = observed_ts.dropna()
    return observed_ts