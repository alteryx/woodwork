import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

import woodwork as ww
from woodwork.data_column import infer_logical_type
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
    Timedelta,
    WholeNumber
)


def pd_to_dask(series):
    return dd.from_pandas(series, npartitions=2)


# Integer Inference
@pytest.fixture
def pandas_integers():
    return [
        pd.Series([-1, 2, 1]),
        pd.Series([-1, 0, 5]),
    ]


@pytest.fixture
def dask_integers(pandas_integers):
    return [pd_to_dask(series) for series in pandas_integers]


@pytest.fixture(params=['pandas_integers', 'dask_integers'])
def integers(request):
    return request.getfixturevalue(request.param)


def test_integer_inference(integers):
    dtypes = ['int8', 'int16', 'int32', 'int64', 'intp', 'int', 'Int64']
    for series in integers:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Integer


# WholeNumber Inference
@pytest.fixture
def pandas_whole_nums():
    return [
        pd.Series([0, 1, 5]),
        pd.Series([2, 3, 5]),
    ]


@pytest.fixture
def dask_whole_nums(pandas_whole_nums):
    return [pd_to_dask(series) for series in pandas_whole_nums]


@pytest.fixture(params=['pandas_whole_nums', 'dask_whole_nums'])
def whole_nums(request):
    return request.getfixturevalue(request.param)


def test_whole_number_inference(whole_nums):
    dtypes = ['int8', 'int16', 'int32', 'int64', 'uint8',
              'uint16', 'uint32', 'uint64', 'intp', 'uintp', 'int', 'Int64']
    for series in whole_nums:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == WholeNumber


# Double Inference
@pytest.fixture
def pandas_doubles():
    return [
        pd.Series([-1, 2.0, 1]),
        pd.Series([1, np.nan, 1])
    ]


@pytest.fixture
def dask_doubles(pandas_doubles):
    return [pd_to_dask(series) for series in pandas_doubles]


@pytest.fixture(params=['pandas_doubles', 'dask_doubles'])
def doubles(request):
    return request.getfixturevalue(request.param)


def test_double_inference(doubles):
    dtypes = ['float', 'float32', 'float64', 'float_']
    for series in doubles:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Double


# Boolean Inference
@pytest.fixture
def pandas_bools():
    return [
        pd.Series([True, False, True]),
        pd.Series([True, False, np.nan]),
    ]


@pytest.fixture
def dask_bools(pandas_bools):
    return [pd_to_dask(series) for series in pandas_bools]


@pytest.fixture(params=['pandas_bools', 'dask_bools'])
def bools(request):
    return request.getfixturevalue(request.param)


def test_boolean_inference(bools):
    dtypes = ['bool', 'boolean']
    for series in bools:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Boolean


# Datetime Inference
@pytest.fixture
def pandas_datetimes():
    return [
        pd.Series(['3/11/2000', '3/12/2000', '3/13/2000']),
        pd.Series(['3/11/2000', '3/12/2000', np.nan]),
    ]


@pytest.fixture
def dask_datetimes(pandas_datetimes):
    return [pd_to_dask(series) for series in pandas_datetimes]


@pytest.fixture(params=['pandas_datetimes', 'dask_datetimes'])
def datetimes(request):
    return request.getfixturevalue(request.param)


def test_datetime_inference(datetimes):
    dtypes = ['object', 'string', 'datetime64[ns]']
    for series in datetimes:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Datetime


# Categorical Inference
@pytest.fixture
def pandas_categories():
    return [
        pd.Series(['a', 'b', 'a']),
        pd.Series(['1', '2', '1']),
        pd.Series(['a', 'b', np.nan]),
        pd.Series([1, 2, 1])
    ]


@pytest.fixture
def dask_categories(pandas_categories):
    return [pd_to_dask(series) for series in pandas_categories]


@pytest.fixture(params=['pandas_categories', 'dask_categories'])
def categories(request):
    return request.getfixturevalue(request.param)


def test_categorical_inference(categories):
    dtypes = ['object', 'string', 'category']
    for series in categories:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical


# Timedelta Inference
@pytest.fixture
def pandas_timedeltas():
    return [
        pd.Series(pd.to_timedelta(range(3), unit='s')),
        pd.Series([pd.to_timedelta(1, unit='s'), np.nan])
    ]


@pytest.fixture
def dask_timedeltas(pandas_timedeltas):
    return [pd_to_dask(series) for series in pandas_timedeltas]


@pytest.fixture(params=['pandas_timedeltas', 'dask_timedeltas'])
def timedeltas(request):
    return request.getfixturevalue(request.param)


def test_timedelta_inference(timedeltas):
    dtypes = ['timedelta64[ns]']
    for series in timedeltas:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Timedelta


# NaturalLanguage Inference
@pytest.fixture
def pandas_strings():
    return [
        pd.Series(['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown']),
    ]


@pytest.fixture
def dask_strings(pandas_strings):
    return [pd_to_dask(series) for series in pandas_strings]


@pytest.fixture(params=['pandas_strings', 'dask_strings'])
def strings(request):
    return request.getfixturevalue(request.param)


def test_natural_language_inference(strings):
    dtypes = ['object', 'string']
    for series in strings:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == NaturalLanguage


# NaturalLanguage Inference with Threshold
@pytest.fixture
def pandas_long_strings():
    natural_language_series = pd.Series([
        '01234567890123456789',
        '01234567890123456789',
        '01234567890123456789'])
    category_series = pd.Series([
        '0123456789012345678',
        '0123456789012345678',
        '0123456789012345678'])
    return [natural_language_series, category_series]


@pytest.fixture
def dask_long_strings(pandas_long_strings):
    return [pd_to_dask(series) for series in pandas_long_strings]


@pytest.fixture(params=['pandas_long_strings', 'dask_long_strings'])
def long_strings(request):
    return request.getfixturevalue(request.param)


def test_natural_language_inference_with_threshhold(long_strings):
    dtypes = ['object', 'string']

    ww.config.set_option('natural_language_threshold', 19)
    for dtype in dtypes:
        inferred_type = infer_logical_type(long_strings[0].astype(dtype))
        assert inferred_type == NaturalLanguage
        inferred_type = infer_logical_type(long_strings[1].astype(dtype))
        assert inferred_type == Categorical
    ww.config.reset_option('natural_language_threshold')


# Inference with pd.NA values
@pytest.fixture
def pandas_pdnas():
    return [
        pd.Series(['Mr. John Doe', pd.NA, 'James Brown']).astype('string'),
        pd.Series([1, pd.NA, -2]).astype('Int64'),
        pd.Series([1, pd.NA, 2]).astype('Int64'),
        pd.Series([True, pd.NA, False]).astype('boolean'),
    ]


@pytest.fixture
def dask_pdnas(pandas_pdnas):
    return [pd_to_dask(series) for series in pandas_pdnas]


@pytest.fixture(params=['pandas_pdnas', 'dask_pdnas'])
def pdnas(request):
    return request.getfixturevalue(request.param)


def test_pdna_inference(pdnas):
    expected_logical_types = [
        NaturalLanguage,
        Integer,
        WholeNumber,
        Boolean,
    ]

    for index, series in enumerate(pdnas):
        inferred_type = infer_logical_type(series)
        assert inferred_type == expected_logical_types[index]
