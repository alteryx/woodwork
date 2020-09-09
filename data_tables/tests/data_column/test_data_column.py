import numpy as np
import pandas as pd
import pytest

from data_tables.data_column import DataColumn, infer_logical_type
from data_tables.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
    Timedelta,
    WholeNumber
)


def test_data_column_init(sample_series):
    data_col = DataColumn(sample_series)
    assert data_col.series is sample_series
    assert data_col.name == sample_series.name
    assert data_col.logical_type == Categorical
    assert data_col.semantic_types == {}


def test_data_column_init_with_logical_type(sample_series):
    data_col = DataColumn(sample_series, NaturalLanguage)
    assert data_col.logical_type == NaturalLanguage


def test_data_column_init_with_semantic_types(sample_series):
    semantic_types = {
        'index': {},
        'tag2': {'key': 'value'},
    }
    data_col = DataColumn(sample_series, semantic_types=semantic_types)
    assert data_col.semantic_types == semantic_types


def test_data_column_with_alternate_semantic_types_input(sample_series):
    semantic_types = 'index'
    data_col = DataColumn(sample_series, semantic_types=semantic_types)
    assert data_col.semantic_types == {'index': {}}

    semantic_types = ['index', 'numeric']
    data_col = DataColumn(sample_series, semantic_types=semantic_types)
    assert data_col.semantic_types == {'index': {}, 'numeric': {}}

    semantic_types = {'index': None}
    data_col = DataColumn(sample_series, semantic_types=semantic_types)
    assert data_col.semantic_types == {'index': {}}

    semantic_types = {'tag': {'tag_additional': 'value'}}
    data_col = DataColumn(sample_series, semantic_types=semantic_types)
    assert data_col.semantic_types == {'tag': {'tag_additional': 'value'}}


def test_invalid_logical_type(sample_series):
    error_message = "Invalid logical type specified for 'sample_series'"
    with pytest.raises(TypeError, match=error_message):
        DataColumn(sample_series, int)


def test_semantic_type_errors(sample_series):
    error_message = "semantic_types must be a string, list or dictionary"
    with pytest.raises(TypeError, match=error_message):
        DataColumn(sample_series, semantic_types=int)

    error_message = "Semantic types must be specified as strings"
    with pytest.raises(TypeError, match=error_message):
        DataColumn(sample_series, semantic_types={1: {}})
    with pytest.raises(TypeError, match=error_message):
        DataColumn(sample_series, semantic_types=['index', 1])

    error_message = "Additional semantic type data must be specified in a dictionary"
    with pytest.raises(TypeError, match=error_message):
        DataColumn(sample_series, semantic_types={'index': 1})


def test_integer_inference():
    series_list = [
        pd.Series([-1, 2, 1]),
        pd.Series([-1, 0, 5]),
    ]
    dtypes = ['int8', 'int16', 'int32', 'int64', 'intp', 'int']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Integer


def test_whole_number_inference():
    series_list = [
        pd.Series([0, 1, 5]),
        pd.Series([2, 3, 5]),
    ]
    dtypes = ['int8', 'int16', 'int32', 'int64', 'uint8',
              'uint16', 'uint32', 'uint64', 'intp', 'uintp', 'int']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == WholeNumber


def test_double_inference():
    series_list = [
        pd.Series([-1, 2.0, 1]),
        pd.Series([1, np.nan, 1])
    ]
    dtypes = ['float', 'float32', 'float64', 'float_']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Double


def test_boolean_inference():
    series_list = [
        pd.Series([True, False, True]),
        pd.Series([True, False, np.nan]),
    ]
    dtypes = ['bool']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Boolean


def test_datetime_inference():
    series_list = [
        pd.Series(['3/11/2000', '3/12/2000', '3/13/2000']),
        pd.Series(['3/11/2000', '3/12/2000', np.nan]),
    ]
    dtypes = ['object', 'string', 'datetime64[ns]']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Datetime


def test_categorical_inference():
    series_list = [
        pd.Series(['a', 'b', 'a']),
        pd.Series(['1', '2', '1']),
        pd.Series(['a', 'b', np.nan]),
        pd.Series([1, 2, 1])
    ]
    dtypes = ['object', 'string', 'category']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical


def test_timedelta_inference():
    series_list = [
        pd.Series(pd.to_timedelta(range(3), unit='s')),
        pd.Series([pd.to_timedelta(1, unit='s'), np.nan])
    ]
    dtypes = ['timedelta64[ns]']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Timedelta


def test_natural_language_inference():
    series_list = [
        pd.Series(['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown']),
    ]
    dtypes = ['object', 'string']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == NaturalLanguage


def test_data_column_repr(sample_series):
    data_col = DataColumn(sample_series)
    assert data_col.__repr__() == "<DataColumn: sample_series (Physical Type = object) (Logical Type = Categorical) (Semantic Tags = {})>"
