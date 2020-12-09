
import numpy as np
import pandas as pd
import pytest

from woodwork.logical_types import (
    Categorical,
    CountryCode,
    Double,
    Integer,
    NaturalLanguage
)
from woodwork.type_sys.inference_functions import (
    categorical_func,
    double_func,
    integer_func
)
from woodwork.type_sys.type_system import TypeSystem


def pd_to_dask(series):
    dd = pytest.importorskip('dask.dataframe', reason='Dask not installed, skipping')
    return dd.from_pandas(series, npartitions=2)


def pd_to_koalas(series):
    ks = pytest.importorskip('databricks.koalas', reason='Koalas not installed, skipping')
    return ks.from_pandas(series)


# Integer Inference Fixtures
@pytest.fixture
def pandas_integers():
    return [
        pd.Series([-1, 2, 1, 7]),
        pd.Series([-1, 0, 5, 3]),
    ]


@pytest.fixture
def dask_integers(pandas_integers):
    return [pd_to_dask(series) for series in pandas_integers]


@pytest.fixture
def koalas_integers(pandas_integers):
    return [pd_to_koalas(series) for series in pandas_integers]


@pytest.fixture(params=['pandas_integers', 'dask_integers', 'koalas_integers'])
def integers(request):
    return request.getfixturevalue(request.param)


# Double Inference Fixtures
@pytest.fixture
def pandas_doubles():
    return [
        pd.Series([-1, 2.5, 1, 7]),
        pd.Series([1.5, np.nan, 1, 3])
    ]


@pytest.fixture
def dask_doubles(pandas_doubles):
    return [pd_to_dask(series) for series in pandas_doubles]


@pytest.fixture
def koalas_doubles(pandas_doubles):
    return [pd_to_koalas(series) for series in pandas_doubles]


@pytest.fixture(params=['pandas_doubles', 'dask_doubles', 'koalas_doubles'])
def doubles(request):
    return request.getfixturevalue(request.param)


# Boolean Inference Fixtures
@pytest.fixture
def pandas_bools():
    return [
        pd.Series([True, False, True, True]),
        pd.Series([True, np.nan, True, True]),
    ]


@pytest.fixture
def dask_bools(pandas_bools):
    return [pd_to_dask(series) for series in pandas_bools]


@pytest.fixture
def koalas_bools(pandas_bools):
    return [pd_to_koalas(series) for series in pandas_bools]


@pytest.fixture(params=['pandas_bools', 'dask_bools', 'koalas_bools'])
def bools(request):
    return request.getfixturevalue(request.param)


# Datetime Inference Fixtures
@pytest.fixture
def pandas_datetimes():
    return [
        pd.Series(['3/11/2000', '3/12/2000', '3/13/2000', '3/14/2000']),
        pd.Series(['3/11/2000', np.nan, '3/13/2000', '3/14/2000']),
    ]


@pytest.fixture
def dask_datetimes(pandas_datetimes):
    return [pd_to_dask(series) for series in pandas_datetimes]


@pytest.fixture
def koalas_datetimes(pandas_datetimes):
    return [pd_to_koalas(series) for series in pandas_datetimes]


@pytest.fixture(params=['pandas_datetimes', 'dask_datetimes', 'koalas_datetimes'])
def datetimes(request):
    return request.getfixturevalue(request.param)


# Categorical Inference Fixtures
@pytest.fixture
def pandas_categories():
    return [
        pd.Series(['a', 'b', 'a', 'b']),
        pd.Series(['1', '2', '1', '2']),
        pd.Series(['a', np.nan, 'b', 'b']),
        pd.Series([1, 2, 1, 2])
    ]


@pytest.fixture
def dask_categories(pandas_categories):
    return [pd_to_dask(series) for series in pandas_categories]


@pytest.fixture
def koalas_categories(pandas_categories):
    return [pd_to_koalas(series) for series in pandas_categories]


@pytest.fixture(params=['pandas_categories', 'dask_categories', 'koalas_categories'])
def categories(request):
    return request.getfixturevalue(request.param)


# Timedelta Inference Fixtures
@pytest.fixture
def pandas_timedeltas():
    return [
        pd.Series(pd.to_timedelta(range(4), unit='s')),
        pd.Series([pd.to_timedelta(1, unit='s'), np.nan])
    ]


@pytest.fixture
def dask_timedeltas(pandas_timedeltas):
    return [pd_to_dask(series) for series in pandas_timedeltas]


@pytest.fixture(params=['pandas_timedeltas', 'dask_timedeltas'])
def timedeltas(request):
    return request.getfixturevalue(request.param)


# NaturalLanguage Inference Fixtures
@pytest.fixture
def pandas_strings():
    return [
        pd.Series(['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown', 'Ms. Paige Turner']),
    ]


@pytest.fixture
def dask_strings(pandas_strings):
    return [pd_to_dask(series) for series in pandas_strings]


@pytest.fixture
def koalas_strings(pandas_strings):
    return [pd_to_koalas(series) for series in pandas_strings]


@pytest.fixture(params=['pandas_strings', 'dask_strings', 'koalas_strings'])
def strings(request):
    return request.getfixturevalue(request.param)


# NaturalLanguage Inference with Threshold
@pytest.fixture
def pandas_long_strings():
    natural_language_series = pd.Series([
        '01234567890123456789',
        '01234567890123456789',
        '01234567890123456789',
        '01234567890123456789'])
    category_series = pd.Series([
        '0123456789012345678',
        '0123456789012345678',
        '0123456789012345678',
        '0123456789012345678'])
    return [natural_language_series, category_series]


@pytest.fixture
def dask_long_strings(pandas_long_strings):
    return [pd_to_dask(series) for series in pandas_long_strings]


@pytest.fixture
def koalas_long_strings(pandas_long_strings):
    return [pd_to_koalas(series) for series in pandas_long_strings]


@pytest.fixture(params=['pandas_long_strings', 'dask_long_strings', 'koalas_long_strings'])
def long_strings(request):
    return request.getfixturevalue(request.param)


# pd.NA Inference Fixtures
@pytest.fixture
def pandas_pdnas():
    return [
        pd.Series(['Mr. John Doe', pd.NA, 'James Brown', 'Ms. Paige Turner']).astype('string'),
        pd.Series([1, pd.NA, 2, 3]).astype('Int64'),
        pd.Series([True, pd.NA, False, True]).astype('boolean'),
    ]


@pytest.fixture
def dask_pdnas(pandas_pdnas):
    return [pd_to_dask(series) for series in pandas_pdnas]


@pytest.fixture(params=['pandas_pdnas', 'dask_pdnas'])
def pdnas(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def default_inference_functions():
    return {
        Double: double_func,
        Integer: integer_func,
        Categorical: categorical_func,
        CountryCode: None,
        NaturalLanguage: None,
    }


@pytest.fixture
def default_relationships():
    return [(Double, Integer), (Categorical, CountryCode)]


@pytest.fixture
def type_sys(default_inference_functions, default_relationships):
    return TypeSystem(inference_functions=default_inference_functions,
                      relationships=default_relationships,
                      default_type=NaturalLanguage)
