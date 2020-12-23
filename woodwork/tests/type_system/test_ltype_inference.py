
import woodwork as ww
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    LogicalType,
    NaturalLanguage,
    Timedelta
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
    'float32',  # compatible starting with 1.4.0
    'float_',
    'object',
    'datetime64[ns]',  # compatible starting with 1.4.0
    'category',
    'string',
    'boolean',
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
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert inferred_type == Integer


def test_double_inference(doubles):
    dtypes = ['float', 'float32', 'float64', 'float_']
    if ks and isinstance(doubles[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    for series in doubles:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert inferred_type == Double


def test_boolean_inference(bools):
    dtypes = ['bool', 'boolean']
    if ks and isinstance(bools[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)
    for series in bools:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert inferred_type == Boolean


def test_datetime_inference(datetimes):
    dtypes = ['object', 'string', 'datetime64[ns]']
    if ks and isinstance(datetimes[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    for series in datetimes:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert inferred_type == Datetime


def test_categorical_inference(categories):
    dtypes = ['object', 'string', 'category']
    if ks and isinstance(categories[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)
    for series in categories:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical


def test_categorical_integers_inference(integers):
    ww.config.set_option('numeric_categorical_threshold', 10)
    dtypes = ['int8', 'int16', 'int32', 'int64', 'intp', 'int', 'Int64']
    if ks and isinstance(integers[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)
    for series in integers:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical
    ww.config.reset_option('numeric_categorical_threshold')


def test_categorical_double_inference(doubles):
    ww.config.set_option('numeric_categorical_threshold', 10)
    dtypes = ['float', 'float32', 'float64', 'float_']
    if ks and isinstance(doubles[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)
    for series in doubles:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical
    ww.config.reset_option('numeric_categorical_threshold')


def test_timedelta_inference(timedeltas):
    dtypes = ['timedelta64[ns]']
    for series in timedeltas:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert inferred_type == Timedelta


def test_natural_language_inference(strings):
    dtypes = ['object', 'string']
    if ks and isinstance(strings[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    for series in strings:
        for dtype in dtypes:
            inferred_type = ww.type_system.infer_logical_type(series.astype(dtype))
            assert inferred_type == NaturalLanguage


def test_natural_language_inference_with_threshhold(long_strings):
    dtypes = ['object', 'string']
    if ks and isinstance(long_strings[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    ww.config.set_option('natural_language_threshold', 19)
    for dtype in dtypes:
        inferred_type = ww.type_system.infer_logical_type(long_strings[0].astype(dtype))
        assert inferred_type == NaturalLanguage
        inferred_type = ww.type_system.infer_logical_type(long_strings[1].astype(dtype))
        assert inferred_type == Categorical
    ww.config.reset_option('natural_language_threshold')


def test_pdna_inference(pdnas):
    expected_logical_types = [
        NaturalLanguage,
        Integer,
        Boolean,
    ]

    for index, series in enumerate(pdnas):
        inferred_type = ww.type_system.infer_logical_type(series)
        assert inferred_type == expected_logical_types[index]


def test_updated_ltype_inference(integers, type_sys):
    inference_fn = type_sys.inference_functions[ww.logical_types.Integer]
    type_sys.remove_type(ww.logical_types.Integer)

    class Integer(LogicalType):
        pandas_dtype = 'string'

    type_sys.add_type(Integer, inference_function=inference_fn)

    dtypes = ['int8', 'int16', 'int32', 'int64', 'intp', 'int', 'Int64']
    if ks and isinstance(integers[0], ks.Series):
        dtypes = get_koalas_dtypes(dtypes)

    for series in integers:
        for dtype in dtypes:
            inferred_type = type_sys.infer_logical_type(series.astype(dtype))
            assert inferred_type == Integer
            assert inferred_type.pandas_dtype == 'string'
