from woodwork.logical_types import LatLong
from woodwork.type_sys.utils import _get_ltype_class


def clean_latlong(dataframe):
    """Convert latlong tuples to strings for parquet, arrow and feather file format.
    Attempting to serialize with tuples present results in an error"""
    latlong_columns = [
        col_name
        for col_name, col in dataframe.ww.columns.items()
        if _get_ltype_class(col.logical_type) == LatLong
    ]
    if len(latlong_columns) > 0:
        dataframe = dataframe.ww.copy()
        dataframe[latlong_columns] = dataframe[latlong_columns].astype(str)

    return dataframe
