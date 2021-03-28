
from minimum_dependency_generator import find_min_requirement
from packaging.specifiers import Specifier

import sys

def test_lower_bound():
    mininum_package = find_min_requirement('plotly>=4.14.0')
    verify_mininum(mininum_package, 'plotly', '4.14.0')


def test_lower_upper_bound():
    mininum_package = find_min_requirement('xgboost>=0.82,<1.3.0')
    verify_mininum(mininum_package, 'xgboost', '0.82')
    mininum_package = find_min_requirement('dask>=2.30.0,<2012.12.0')
    verify_mininum(mininum_package, 'dask', '2.30.0')


def test_spacing():
    mininum_package = find_min_requirement('statsmodels >= 0.12.2')
    verify_mininum(mininum_package, 'statsmodels', '0.12.2')


def test_marker():
    mininum_package = find_min_requirement('sktime>=0.5.3;python_version<"3.9"')
    verify_mininum(mininum_package, 'sktime', '0.5.3')

def test_upper_bound():
    mininum_package = find_min_requirement('pyzmq<22.0.0')
    verify_mininum(mininum_package, 'pyzmq', '2.0.7')
    mininum_package = find_min_requirement('dask<2012.12.0')
    verify_mininum(mininum_package, 'dask', '0.2.0')


def test_not_bound():
    mininum_package = find_min_requirement('colorama')
    verify_mininum(mininum_package, 'colorama', '0.1')


def test_bound():
    mininum_package = find_min_requirement('woodwork==0.0.11')
    verify_mininum(mininum_package, 'woodwork', '0.0.11')


def test_extra_requires():
    mininum_package = find_min_requirement('dask[dataframe]>=2.30.0,<2012.12.0')
    verify_mininum(mininum_package, 'dask', '2.30.0', required_extra='dataframe')


def test_comments():
    mininum_package = find_min_requirement('pyspark>=3.0.0 ; python_version!=\'3.9\' # comment here')
    verify_mininum(mininum_package, 'pyspark', '3.0.0')


def test_complex_bound():
    mininum_package = find_min_requirement('pandas>=0.24.1,<2.0.0,!=1.1.0,!=1.1.1')
    verify_mininum(mininum_package, 'pandas', '0.24.1')

def test_wrong_python_env():
    mininum_package = find_min_requirement('ipython==7.16.0; python_version==\'3.4\'')
    assert mininum_package is None
    python_version = str(sys.version_info.major) + '.' + str(sys.version_info.minor)
    mininum_package = find_min_requirement('ipython==7.16.0; python_version==\'' + python_version + '\'')
    verify_mininum(mininum_package, 'ipython', '7.16.0')


def verify_mininum(mininum_package, required_package_name, required_mininum_version,
                   operator='==', required_extra=None):
    assert mininum_package.name == required_package_name
    assert mininum_package.specifier == Specifier(operator + required_mininum_version)
    if required_extra:
        mininum_package.extras == {required_extra}