def get_ranges(idx_series):
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
