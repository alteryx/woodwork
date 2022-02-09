import pandas as pd


def generate_estimated_timeseries(freq_df, inferred_freq):
    series = freq_df[inferred_freq]

    start = series[series].index.min()
    end = series[series].index.max()

    return pd.date_range(start, end, freq=inferred_freq)
