import pandas as pd


def _get_histogram_values(series, bins=10):
    """Get the histogram for a given numeric column.

    Args:
        series (pd.Series): data to use for histogram
        bins (int): the number of bins to use for the histogram

    Returns:
        histogram (list(dict)): a list of dictionary with keys `bins` and
            `frequency`
    """
    if str(series.dtype) in [
        "boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Float32",
        "Float64",
        "float16",
        "float32",
        "float64",
        "float128",
    ]:
        series = series.astype(float)
    values = (
        pd.cut(x=series.to_numpy(), bins=bins, duplicates="drop")
        .value_counts()
        .sort_index()
    )
    df = values.reset_index()
    df.columns = ["bins", "frequency"]
    results = []
    for _, row in df.iterrows():
        results.append(
            {
                "bins": [row["bins"].left, row["bins"].right],
                "frequency": row["frequency"],
            },
        )

    return results
