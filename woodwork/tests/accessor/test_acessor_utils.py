import pytest

from woodwork.accessor_utils import init_series
from woodwork.logical_types import Categorical, Datetime
from woodwork.tests.testing_utils import xfail_dask_and_koalas


def test_init_series_valid_conversion_specified_ltype(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = init_series(sample_series, logical_type='categorical')
    assert series is not sample_series
    assert series.dtype == 'category'
    assert series.ww.logical_type == Categorical
    assert series.ww.semantic_tags == {'category'}


def test_init_series_valid_conversion_inferred_ltype(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = init_series(sample_series)
    assert series is not sample_series
    assert series.dtype == 'category'
    assert series.ww.logical_type == Categorical
    assert series.ww.semantic_tags == {'category'}


def test_init_series_with_datetime(sample_datetime_series):
    xfail_dask_and_koalas(sample_datetime_series)
    series = init_series(sample_datetime_series, logical_type='datetime')
    assert series.dtype == 'datetime64[ns]'
    assert series.ww.logical_type == Datetime


def test_init_series_error_on_invalid_conversion(sample_series):
    xfail_dask_and_koalas(sample_series)
    error_message = "Error converting datatype for sample_series from type object to type Int64. " \
        "Please confirm the underlying data is consistent with logical type Integer."
    with pytest.raises(TypeError, match=error_message):
        init_series(sample_series, logical_type='integer')
