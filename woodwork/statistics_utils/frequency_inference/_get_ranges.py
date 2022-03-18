def _get_ranges(idx_series):
    """Generates ranges of contiguous values.

    For example, if input idx_series = [10,11,15,19,20,21,22,23,24,30].

    The output ranges would be:

    [
        (10, 11),
        (15, 15),
        (19, 24),
        (30, 30)
    ]

    Args:
        idx_series (pd.Series): a pandas integer index

    Returns:
        (List(tuple)): a list of tuples, where the tuple is the (starting value, ending value) inclusive
    """
    start_idx = idx_series[0]
    last_idx = idx_series[0]
    ranges = []
    for idx in idx_series[1:]:
        if (idx - last_idx) > 1:
            ranges.append((start_idx, last_idx))
            start_idx = idx

        last_idx = idx

    ranges.append((start_idx, idx_series[-1]))
    return ranges
