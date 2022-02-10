import pandas as pd

from woodwork.logical_types import Datetime, Timedelta


def _infer_temporal_frequencies(dataframe, temporal_columns=None):
    """Infers the observation frequency (daily, biweekly, yearly, etc) of each temporal column
            in the DataFrame. Temporal columns are ones with the logical type Datetime or Timedelta.

    Args:
        dataframe (DataFrame): The DataFrame for which frequncies should be inferred.
        temporal_columns (list[str], optional): Columns for which frequencies should be inferred. Must be columns
            that are present in the DataFrame and are temporal in nature. Defaults to None. If not
            specified, all temporal columns will have their frequencies inferred.

    Returns:
        (dict): A dictionary where each key is a temporal column from the DataFrame, and the
            value is its observation frequency represented as a pandas offset alias string (D, M, Y, etc.)
            or None if no uniform frequency was present in the data.
    """
    logical_types = dataframe.ww.logical_types

    if temporal_columns is None:
        temporal_columns = [
            col
            for col, ltype in logical_types.items()
            if isinstance(ltype, (Datetime, Timedelta))
        ]
    else:
        for col in temporal_columns:
            if col not in dataframe:
                raise ValueError(f"Column {col} not found in dataframe.")
            ltype = logical_types[col]
            if not isinstance(ltype, (Datetime, Timedelta)):
                raise TypeError(
                    f"Cannot determine frequency for column {col} with logical type {ltype}"
                )

    return {col: pd.infer_freq(dataframe[col]) for col in temporal_columns}
