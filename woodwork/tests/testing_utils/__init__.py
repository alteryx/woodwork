# flake8: noqa
import pytest
from .table_utils import (
    check_empty_box_plot_dict,
    is_property,
    is_public_method,
    mi_between_cols,
    to_pandas,
    validate_subset_schema,
)


def pd_to_dask(series):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    return dd.from_pandas(series, npartitions=2)


def pd_to_spark(series):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(convert_tuples_to_lists(series))


def convert_tuples_to_lists(series):
    def apply_func(value):
        if type(value) is tuple:
            return list(value)
        return value

    return series.apply(apply_func)
