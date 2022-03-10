import pandas as pd


def _generate_estimated_timeseries(alias_dict):
    """Generates Estimated Timeseries:

    Args:
        alias_dict (dict): The alias dictionary has the following properties:

        - alias: the pandas frequency alias
        - min_dt: the minimum timestamp for this alias
        - max_dt: the maximum timestamp for this alias

    Returns:
        (DatetimeIndex): The estimated datetime index
    """
    alias = alias_dict["alias"]
    start = alias_dict["min_dt"]
    end = alias_dict["max_dt"]

    return pd.date_range(start, end, freq=alias)
