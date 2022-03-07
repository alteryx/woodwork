import pandas as pd


def _generate_estimated_timeseries(alias_dict):
    alias = alias_dict["alias"]
    start = alias_dict["min_dt"]
    end = alias_dict["max_dt"]

    return pd.date_range(start, end, freq=alias)
