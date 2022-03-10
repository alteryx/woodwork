def _get_ranges(idx_series):
    """Generates ranges of contiguous values:

    Args:
        alias_dict (dict): The alias dictionary has the following properties:

        - alias: the pandas frequency alias
        - min_dt: the minimum timestamp for this alias
        - max_dt: the maximum timestamp for this alias

    Returns:
        (DatetimeIndex): The estimated datetime index
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
