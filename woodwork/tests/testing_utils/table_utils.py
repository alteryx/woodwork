import numpy as np
import pandas as pd

from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def validate_subset_schema(subset_schema, schema):
    assert subset_schema.name == schema.name
    for subset_col_name, subset_col in subset_schema.columns.items():
        assert subset_col_name in schema.columns
        col = schema.columns[subset_col_name]
        assert subset_col.logical_type == col.logical_type
        assert subset_col.semantic_tags == col.semantic_tags


def mi_between_cols(col1, col2, df):
    mi_series = df.loc[df['column_1'] == col1].loc[df['column_2'] == col2]['mutual_info']

    if len(mi_series) == 0:
        mi_series = df.loc[df['column_1'] == col2].loc[df['column_2'] == col1]['mutual_info']

    return mi_series.iloc[0]


def to_pandas(df, index=None, sort_index=False):
    """Testing util to convert dataframes to pandas. If a pandas dataframe is passed in, just returns the dataframe.

    Returns:
        Pandas DataFrame
    """
    if isinstance(df, (pd.DataFrame, pd.Series, pd.Index)):
        return df

    if dd and isinstance(df, (dd.DataFrame, dd.Series, dd.Index)):
        pd_df = df.compute()

    if ks and isinstance(df, (ks.DataFrame, ks.Series, ks.Index)):
        pd_df = df.to_pandas()

    if index:
        pd_df = pd_df.set_index(index, drop=False)
    if sort_index:
        pd_df = pd_df.sort_index()

    return pd_df


def is_public_method(class_to_check, name):
    """Determine if the specified name is a public method on a class"""
    if hasattr(class_to_check, name) and name[0] != '_':
        if callable(getattr(class_to_check, name)):
            return True
    return False


def is_property(class_to_check, name):
    """Determine if the specified name is a property on a class"""
    if hasattr(class_to_check, name) and isinstance(getattr(class_to_check, name), property):
        return True
    return False


def check_empty_box_plot_dict(box_plot_dict):
    assert np.isnan(box_plot_dict['low_bound'])
    assert np.isnan(box_plot_dict['high_bound'])
    assert len(box_plot_dict['quantiles']) == 5
    assert all([np.isnan(elt) for elt in box_plot_dict['quantiles'].values()])
    assert len(box_plot_dict['low_values']) == 0
    assert len(box_plot_dict['high_values']) == 0
