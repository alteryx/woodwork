from pyarrow.lib import ArrowNotImplementedError


def _get_mode(series):
    """Get the mode value for a series"""
    try:
        mode_values = series.mode()
    except ArrowNotImplementedError:
        mode_values = series.astype("string").mode()
    if len(mode_values) > 0:
        return mode_values[0]
    return None
