import sys
import tempfile

import pytest
from packaging.specifiers import Specifier

from .minimum_dependency_generator import (
    find_min_requirement,
    write_min_requirements
)


@pytest.fixture(scope='session', autouse=True)
def pandas_dep():
    return 'pandas>=0.24.1,<2.0.0,!=1.1.0,!=1.1.1'


@pytest.fixture(scope='session', autouse=True)
def woodwork_dep():
    return 'woodwork==0.0.11'


@pytest.fixture(scope='session', autouse=True)
def dask_dep():
    return 'dask[dataframe]>=2.30.0,<2012.12.0'


@pytest.fixture(scope='session', autouse=True)
def ploty_dep():
    return 'plotly>=4.14.0'


@pytest.fixture(scope='session', autouse=True)
def numpy_lower():
    return 'numpy>=1.15.4'


@pytest.fixture(scope='session', autouse=True)
def numpy_upper():
    return 'numpy<1.20.0'


def test_lower_bound(ploty_dep):
    mininum_package = find_min_requirement(ploty_dep)
    verify_mininum(mininum_package, 'plotly', '4.14.0')
    mininum_package = find_min_requirement('plotly>=4.14')
    verify_mininum(mininum_package, 'plotly', '4.14')


def test_lower_upper_bound(dask_dep):
    mininum_package = find_min_requirement('xgboost>=0.82,<1.3.0')
    verify_mininum(mininum_package, 'xgboost', '0.82')
    mininum_package = find_min_requirement(dask_dep)
    verify_mininum(mininum_package, 'dask', '2.30.0', required_extra='dataframe')


def test_spacing():
    mininum_package = find_min_requirement('statsmodels >= 0.12.2')
    verify_mininum(mininum_package, 'statsmodels', '0.12.2')


def test_marker():
    mininum_package = find_min_requirement('sktime>=0.5.3;python_version<"3.9"')
    verify_mininum(mininum_package, 'sktime', '0.5.3')


def test_upper_bound():
    error_text = "Operator does not exist or is an invalid operator. Please specify the mininum version."
    with pytest.raises(ValueError, match=error_text):
        find_min_requirement('xgboost<1.3.0')
    with pytest.raises(ValueError, match=error_text):
        find_min_requirement('colorama')


def test_bound(woodwork_dep):
    mininum_package = find_min_requirement(woodwork_dep)
    verify_mininum(mininum_package, 'woodwork', '0.0.11')


def test_extra_requires():
    mininum_package = find_min_requirement('dask>=2.30.0')
    verify_mininum(mininum_package, 'dask', '2.30.0', required_extra=None)
    mininum_package = find_min_requirement('dask[dataframe]<2012.12.0,>=2.30.0')
    verify_mininum(mininum_package, 'dask', '2.30.0', required_extra='dataframe')


def test_comments():
    mininum_package = find_min_requirement('pyspark>=3.0.0 ; python_version!=\'3.9\' # comment here')
    verify_mininum(mininum_package, 'pyspark', '3.0.0')


def test_complex_bound(pandas_dep):
    mininum_package = find_min_requirement(pandas_dep)
    verify_mininum(mininum_package, 'pandas', '0.24.1')


def test_wrong_python_env():
    mininum_package = find_min_requirement('ipython==7.16.0; python_version==\'3.4\'')
    assert mininum_package is None
    python_version = str(sys.version_info.major) + '.' + str(sys.version_info.minor)
    mininum_package = find_min_requirement('ipython==7.16.0; python_version==\'' + python_version + '\'')
    verify_mininum(mininum_package, 'ipython', '7.16.0')


def test_write_min_requirements(ploty_dep, dask_dep, pandas_dep, woodwork_dep,
                                numpy_upper, numpy_lower):
    min_requirements = []
    requirements_core = '\n'.join([dask_dep, pandas_dep, woodwork_dep, numpy_upper])
    requirements_koalas = '\n'.join([ploty_dep, numpy_lower])
    with tempfile.NamedTemporaryFile(mode="w", suffix='.txt', prefix='out_requirements') as out_f:
        with tempfile.NamedTemporaryFile(mode="w", suffix='.txt', prefix='core_requirements') as core_f:
            with tempfile.NamedTemporaryFile(mode="w", suffix='.txt', prefix='koalas_requirements') as koalas_f:
                core_f.writelines(requirements_core)
                core_f.flush()
                koalas_f.writelines(requirements_koalas)
                koalas_f.flush()
                write_min_requirements(out_f.name, requirements_paths=[core_f.name, koalas_f.name])
        with open(out_f.name) as f:
            min_requirements = f.readlines()
    expected_min_reqs = ['dask[dataframe]==2.30.0', 'pandas==0.24.1', 'woodwork==0.0.11', 'numpy==1.15.4', 'plotly==4.14.0']
    assert len(min_requirements) == len(expected_min_reqs)
    for idx, min_req in enumerate(min_requirements):
        assert expected_min_reqs[idx] == min_req.strip()


def verify_mininum(mininum_package, required_package_name, required_mininum_version,
                   operator='==', required_extra=None):
    assert mininum_package.name == required_package_name
    assert mininum_package.specifier == Specifier(operator + required_mininum_version)
    if required_extra:
        assert mininum_package.extras == {required_extra}
    else:
        assert mininum_package.extras == set()
        extra_chars = ['[', ']']
        not any([x in mininum_package.name for x in extra_chars])
        assert not any([x in required_package_name for x in extra_chars])
