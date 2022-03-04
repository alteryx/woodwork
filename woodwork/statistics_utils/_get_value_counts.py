import numpy as np

from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe


def _get_value_counts(dataframe, ascending=False, top_n=10, dropna=False):
    """Returns a list of dictionaries with counts for the most frequent values in each column (only
        for columns with `category` as a standard tag).


    Args:
        dataframe (pd.DataFrame, dd.DataFrame, ps.DataFrame): Data from which to count values.
        ascending (bool): Defines whether each list of values should be sorted most frequent
            to least frequent value (False), or least frequent to most frequent value (True).
            Defaults to False.

        top_n (int): the number of top values to retrieve. Defaults to 10.

        dropna (bool): determines whether to remove NaN values when finding frequency. Defaults
            to False.

    Returns:
        list(dict): a list of dictionaries for each categorical column with keys `count`
        and `value`.
    """
    val_counts = {}
    valid_cols = [
        col for col, column in dataframe.ww.columns.items() if column.is_categorical
    ]
    data = dataframe[valid_cols]
    is_ks = False
    if _is_dask_dataframe(data):
        data = data.compute()
    if _is_spark_dataframe(data):
        data = data.to_pandas()
        is_ks = True

    for col in valid_cols:
        if dropna and is_ks:
            # Spark categorical columns will have missing values replaced with the string 'None'
            # Replace them with np.nan so dropna work
            datacol = data[col].replace(to_replace="None", value=np.nan)
        else:
            datacol = data[col]
        frequencies = datacol.value_counts(ascending=ascending, dropna=dropna)
        df = frequencies[:top_n].reset_index()
        df.columns = ["value", "count"]
        values = list(df.to_dict(orient="index").values())
        val_counts[col] = values
    return val_counts
