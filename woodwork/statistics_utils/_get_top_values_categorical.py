def _get_top_values_categorical(series, num_x):
    """Get the most frequent values in a pandas Series. Will exclude null values.

    Args:
        column (pd.Series): data to use find most frequent values
        num_x (int): the number of top values to retrieve

    Returns:
        top_list (list(dict)): a list of dictionary with keys `value` and `count`.
            Output is sorted in descending order based on the value counts.
    """
    frequencies = series.value_counts(dropna=True)
    df = frequencies.head(num_x).reset_index()
    df.columns = ["value", "count"]
    df = df.sort_values(["count", "value"], ascending=[False, True])
    value_counts = list(df.to_dict(orient="index").values())
    return value_counts
