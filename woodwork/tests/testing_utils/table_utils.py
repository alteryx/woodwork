import numpy as np
import pandas as pd


def validate_subset_schema(subset_schema, schema):
    assert subset_schema.name == schema.name
    for subset_col_name, subset_col in subset_schema.columns.items():
        assert subset_col_name in schema.columns
        col = schema.columns[subset_col_name]
        assert subset_col.logical_type == col.logical_type
        assert subset_col.semantic_tags == col.semantic_tags


def dep_between_cols(col1, col2, dep_name, df):
    dep_series = df.loc[df["column_1"] == col1].loc[df["column_2"] == col2][dep_name]

    if len(dep_series) == 0:
        dep_series = df.loc[df["column_1"] == col2].loc[df["column_2"] == col1][
            dep_name
        ]

    return dep_series.iloc[0]


def is_public_method(class_to_check, name):
    """Determine if the specified name is a public method on a class"""
    if hasattr(class_to_check, name) and name[0] != "_":
        if callable(getattr(class_to_check, name)):
            return True
    return False


def is_property(class_to_check, name):
    """Determine if the specified name is a property on a class"""
    if hasattr(class_to_check, name) and isinstance(
        getattr(class_to_check, name),
        property,
    ):
        return True
    return False


def check_empty_box_plot_dict(box_plot_dict):
    assert np.isnan(box_plot_dict["low_bound"])
    assert np.isnan(box_plot_dict["high_bound"])
    assert len(box_plot_dict["quantiles"]) == 5
    assert all([np.isnan(elt) for elt in box_plot_dict["quantiles"].values()])
    assert len(box_plot_dict["low_values"]) == 0
    assert len(box_plot_dict["high_values"]) == 0


def assert_schema_equal(left_schema, right_schema, deep=True):
    assert left_schema.name == right_schema.name
    assert left_schema.index == right_schema.index
    assert left_schema.time_index == right_schema.time_index
    assert set(left_schema.columns.keys()) == set(right_schema.columns.keys())
    for col_name in left_schema.columns:
        assert left_schema.columns[col_name].__eq__(
            right_schema.columns[col_name],
            deep=deep,
        )
    if deep:
        assert left_schema.metadata == right_schema.metadata


def _check_close(actual, expected):
    if pd.isnull(expected):
        assert pd.isnull(actual)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-3)
