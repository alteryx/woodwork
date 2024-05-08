# flake8: noqa
import pytest

from woodwork.tests.testing_utils.table_utils import (
    _check_close,
    check_empty_box_plot_dict,
    dep_between_cols,
    is_property,
    is_public_method,
    validate_subset_schema,
)


def convert_tuples_to_lists(series):
    def apply_func(value):
        if type(value) is tuple:
            return list(value)
        return value

    return series.apply(apply_func)
