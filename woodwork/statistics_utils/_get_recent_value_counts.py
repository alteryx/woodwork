def _get_recent_value_counts(column, num_x):
    """Get the the number of occurrences of the x most recent values in a datetime column.

    Args:
        column (pd.Series): data to use find value counts
        num_x (int): the number of values to retrieve

    Returns:
        value_counts (list(dict)): a list of dictionary with keys `value` and
            `count`. Output is sorted in descending order based on the value counts.
    """
    datetimes = getattr(column.dt, "date")
    frequencies = datetimes.value_counts(dropna=False)
    values = frequencies.sort_index(ascending=False)[:num_x]
    df = values.reset_index()
    df.columns = ["value", "count"]
    df = df.sort_values(["count", "value"], ascending=[False, True])
    value_counts = list(df.to_dict(orient="index").values())
    return value_counts
