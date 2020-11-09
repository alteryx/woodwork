import pandas as pd

from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')


def validate_subset_dt(subset_dt, dt):
    assert subset_dt.name == dt.name
    assert len(subset_dt.columns) == len(subset_dt.to_dataframe().columns)
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


def to_pandas(df, index=None):
    '''
    Testing util to convert dataframes to pandas. If a pandas dataframe is passed in, just returns the dataframe.

    Returns:
        Pandas DataFrame
    '''
    if isinstance(df, (pd.DataFrame, pd.Series)):
        return df

    if dd and isinstance(df, (dd.DataFrame, dd.Series)):
        pd_df = df.compute()

    if index:
        pd_df = pd_df.set_index(index, drop=False)

    return pd_df
