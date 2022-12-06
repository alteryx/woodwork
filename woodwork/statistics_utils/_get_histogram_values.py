import pandas as pd

from woodwork.utils import import_or_none

cudf = import_or_none("cudf")


def _get_histogram_values(series, bins=10):
    """Get the histogram for a given numeric column.

    Args:
        series (pd.Series): data to use for histogram
        bins (int): the number of bins to use for the histogram

    Returns:
        histogram (list(dict)): a list of dictionary with keys `bins` and
            `frequency`
    """
    values = pd.cut(series, bins=bins, duplicates="drop").value_counts().sort_index()
    df = values.reset_index()
    df.columns = ["bins", "frequency"]
    results = []
    if isinstance(df, (cudf.DataFrame, cudf.Series)):
        warnings.warn(
            "Can't iterate over cudf objects. Woodwork has automatically converted the cudf to a Pandas dataframe for this calculation. Performance may be worse."
        )
        df_iter = df.to_pandas().iterrows()
    else:
        df_iter = df.iterrows()
    for _, row in df_iter:
        results.append(
            {
                "bins": [row["bins"].left, row["bins"].right],
                "frequency": row["frequency"],
            },
        )

    return results
