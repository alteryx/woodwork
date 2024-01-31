# flake8: noqa
import pytest

from woodwork.tests.testing_utils.table_utils import (
    _check_close,
    check_empty_box_plot_dict,
    concat_dataframe_or_series,
    dep_between_cols,
    is_property,
    is_public_method,
    to_pandas,
    validate_subset_schema,
)


def pd_to_dask(series):
    dask = pytest.importorskip("dask", reason="Dask not installed, skipping")
    dask.config.set({"dataframe.convert-string": False})
    dd = dask.dataframe
    return dd.from_pandas(series, npartitions=1)


def pd_to_spark(series):
    ps = pytest.importorskip("pyspark.pandas", reason="Spark not installed, skipping")
    return ps.from_pandas(convert_tuples_to_lists(series))


def convert_tuples_to_lists(series):
    def apply_func(value):
        if type(value) is tuple:
            return list(value)
        return value

    return series.apply(apply_func)
