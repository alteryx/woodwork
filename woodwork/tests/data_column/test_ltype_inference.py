
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
from woodwork.utils import import_or_none

UNSUPPORTED_KOALAS_DTYPES = [
    'int32',
    'Int64',
    'intp',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'uintp',
    'float32',
    'float_',
    'object',
    'datetime64[ns]',
    'category',
]

ks = import_or_none('databricks.koalas')


def get_koalas_dtypes(dtypes):
    return [dtype for dtype in dtypes if dtype not in UNSUPPORTED_KOALAS_DTYPES]


def test_integer_inference(integers):
    dtypes = ['int8', 'int16', 'int32', 'int64', 'intp', 'int', 'Int64']
    if ks and isinstance(integers[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    for series in integers:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Integer


def test_whole_number_inference(whole_nums):
    dtypes = ['int8', 'int16', 'int32', 'int64', 'uint8',
              'uint16', 'uint32', 'uint64', 'intp', 'uintp', 'int', 'Int64']
    if ks and isinstance(whole_nums[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    for series in whole_nums:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == WholeNumber


def test_double_inference(doubles):
    dtypes = ['float', 'float32', 'float64', 'float_']
    if ks and isinstance(doubles[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    for series in doubles:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Double


def test_boolean_inference(bools):
    dtypes = ['bool', 'boolean']
    for series in bools:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Boolean


def test_datetime_inference(datetimes):
    dtypes = ['object', 'string', 'datetime64[ns]']
    if ks and isinstance(datetimes[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    for series in datetimes:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Datetime


def test_categorical_inference(categories):
    dtypes = ['object', 'string', 'category']
    if ks and isinstance(categories[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)
    for series in categories:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical


def test_categorical_integers_inference(integers):
    ww.config.set_option('numeric_categorical_threshold', 10)
    dtypes = ['int8', 'int16', 'int32', 'int64', 'intp', 'int', 'Int64']
    if ks and isinstance(integers[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)
    for series in integers:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical
    ww.config.reset_option('numeric_categorical_threshold')


def test_categorical_whole_number_inference(whole_nums):
    ww.config.set_option('numeric_categorical_threshold', 10)
    dtypes = ['int8', 'int16', 'int32', 'int64', 'uint8',
              'uint16', 'uint32', 'uint64', 'intp', 'uintp', 'int', 'Int64']
    if ks and isinstance(whole_nums[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)
    for series in whole_nums:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical
    ww.config.reset_option('numeric_categorical_threshold')


def test_categorical_double_inference(doubles):
    ww.config.set_option('numeric_categorical_threshold', 10)
    dtypes = ['float', 'float32', 'float64', 'float_']
    if ks and isinstance(doubles[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)
    for series in doubles:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical
    ww.config.reset_option('numeric_categorical_threshold')


def test_timedelta_inference(timedeltas):
    dtypes = ['timedelta64[ns]']
    for series in timedeltas:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Timedelta


def test_natural_language_inference(strings):
    dtypes = ['object', 'string']
    if ks and isinstance(strings[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    for series in strings:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == NaturalLanguage


def test_natural_language_inference_with_threshhold(long_strings):
    dtypes = ['object', 'string']
    if ks and isinstance(long_strings[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    ww.config.set_option('natural_language_threshold', 19)
    for dtype in dtypes:
        inferred_type = infer_logical_type(long_strings[0].astype(dtype))
        assert inferred_type == NaturalLanguage
        inferred_type = infer_logical_type(long_strings[1].astype(dtype))
        assert inferred_type == Categorical
    ww.config.reset_option('natural_language_threshold')


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
