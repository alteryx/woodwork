import dask.dataframe as dd
import pandas as pd


def validate_subset_dt(subset_dt, dt):
    assert subset_dt.name == dt.name
    assert len(subset_dt.columns) == len(subset_dt.to_pandas().columns)
    for subset_col_name, subset_col in subset_dt.columns.items():
        assert subset_col_name in dt.columns
        col = dt.columns[subset_col_name]
        assert subset_col.logical_type == col.logical_type
        assert subset_col.semantic_tags == col.semantic_tags
        assert subset_col.dtype == col.dtype
        assert to_pandas(subset_col.to_series()).equals(to_pandas(col.to_series()))


def mi_between_cols(col1, col2, df):
    mi_series = df.loc[df['column_1'] == col1].loc[df['column_2'] == col2]['mutual_info']

    if len(mi_series) == 0:
        mi_series = df.loc[df['column_1'] == col2].loc[df['column_2'] == col1]['mutual_info']

    return mi_series.iloc[0]


def to_pandas(df, index=None, sort_index=False, int_index=False):
    '''
    Testing util to convert dataframes to pandas. If a pandas dataframe is passed in, just returns the dataframe.
    Args:
        index (str, optional): column name to set as index, defaults to None
        sort_index (bool, optional): whether to sort the dataframe on the index after setting it, defaults to False
        int_index (bool, optional): Converts computed dask index to Int64Index to avoid errors, defaults to False
    Returns:
        Pandas DataFrame
    '''
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df

    if isinstance(df, (dd.DataFrame, dd.Series)):
        pd_df = df.compute()

    return pd_df
