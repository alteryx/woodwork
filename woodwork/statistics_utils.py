import numpy as np

from woodwork.logical_types import Datetime, LatLong
from woodwork.schema_column import (
    _is_col_boolean,
    _is_col_categorical,
    _is_col_datetime,
    _is_col_numeric
)
from woodwork.type_sys.utils import _get_ltype_class
from woodwork.utils import _get_mode, import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def _get_describe_dict(dataframe, include=None):
    """Calculates statistics for data contained in a DataFrame using Woodwork typing information.

    Args:
        dataframe (pd.DataFrame): DataFrame to be described with Woodwork typing information initialized
        include (list[str or LogicalType], optional): filter for what columns to include in the
        statistics returned. Can be a list of column names, semantic tags, logical types, or a list
        combining any of the three. It follows the most broad specification. Favors logical types
        then semantic tag then column name. If no matching columns are found, an empty DataFrame
        will be returned.

    Returns:
        dict[str -> dict]: A dictionary with a key for each column in the data or for each column
        matching the logical types, semantic tags or column names specified in ``include``, paired
        with a value containing a dictionary containing relevant statistics for that column.
    """
    agg_stats_to_calculate = {
        'category': ["count", "nunique"],
        'numeric': ["count", "max", "min", "nunique", "mean", "std"],
        Datetime: ["count", "max", "min", "nunique", "mean"],
    }
    if include is not None:
        filtered_cols = dataframe.ww._filter_cols(include, col_names=True)
        cols_to_include = [(k, v) for k, v in dataframe.ww.columns.items() if k in filtered_cols]
    else:
        cols_to_include = dataframe.ww.columns.items()

    results = {}

    if dd and isinstance(dataframe, dd.DataFrame):
        df = dataframe.compute()
    elif ks and isinstance(dataframe, ks.DataFrame):
        # Missing values in Koalas will be replaced with 'None' - change them to
        # np.nan so stats are calculated properly
        df = dataframe.to_pandas().replace(to_replace='None', value=np.nan)

        # Any LatLong columns will be using lists, which we must convert
        # back to tuples so we can calculate the mode, which requires hashable values
        latlong_columns = [col_name for col_name, col in dataframe.ww.columns.items() if _get_ltype_class(col['logical_type']) == LatLong]
        df[latlong_columns] = df[latlong_columns].applymap(lambda latlong: tuple(latlong) if latlong else latlong)
    else:
        df = dataframe

    for column_name, column in cols_to_include:
        if 'index' in column['semantic_tags']:
            continue
        values = {}
        logical_type = column['logical_type']
        semantic_tags = column['semantic_tags']
        series = df[column_name]

        # Calculate Aggregation Stats
        if _is_col_categorical(column):
            agg_stats = agg_stats_to_calculate['category']
        elif _is_col_numeric(column):
            agg_stats = agg_stats_to_calculate['numeric']
        elif _is_col_datetime(column):
            agg_stats = agg_stats_to_calculate[Datetime]
        else:
            agg_stats = ["count"]
        values = series.agg(agg_stats).to_dict()

        # Calculate other specific stats based on logical type or semantic tags
        if _is_col_boolean(column):
            values["num_false"] = series.value_counts().get(False, 0)
            values["num_true"] = series.value_counts().get(True, 0)
        elif _is_col_numeric(column):
            quant_values = series.quantile([0.25, 0.5, 0.75]).tolist()
            values["first_quartile"] = quant_values[0]
            values["second_quartile"] = quant_values[1]
            values["third_quartile"] = quant_values[2]

        mode = _get_mode(series)
        # The format of the mode should match its format in the DataFrame
        if ks and isinstance(dataframe, ks.DataFrame) and series.name in latlong_columns:
            mode = list(mode)

        values["nan_count"] = series.isna().sum()
        values["mode"] = mode
        values["physical_type"] = column['dtype']
        values["logical_type"] = logical_type
        values["semantic_tags"] = semantic_tags
        results[column_name] = values
    return results
