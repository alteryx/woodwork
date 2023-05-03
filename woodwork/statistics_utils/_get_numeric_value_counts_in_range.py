from pandas.core.dtypes.common import is_integer_dtype


def _get_numeric_value_counts_in_range(series, _range):
    """Count the number of occurrences of integers present in a series with values defined
    by a range of integers. Null values will be ignored.

    Args:
        series (pd.Series): data from which to determine the number of occurrences of values
        _range (type(range)): sequence of integers defining the values for which counts should be made

    Returns:
        value_counts (list(dict)): a list of dictionaries with keys `value` and
            `count`. Output is sorted in descending order based on the value counts.
    """
    frequencies = series.value_counts(dropna=True)
    value_counts = [
        {
            "value": i if is_integer_dtype(series) else float(i),
            "count": frequencies[i] if i in frequencies else 0,
        }
        for i in _range
    ]
    return sorted(value_counts, key=lambda i: (-i["count"], i["value"]))
