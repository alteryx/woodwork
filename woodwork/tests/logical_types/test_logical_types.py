import re

import pandas as pd
import pytest

from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    LatLong,
    Ordinal
)
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_logical_eq():
    assert Boolean == Boolean
    assert Boolean() == Boolean()
    assert Categorical != Boolean
    assert Datetime != Datetime()
    assert Datetime() == Datetime(datetime_format=None)
    assert Datetime() != Datetime(datetime_format='%Y-%m-%d')


def test_logical_repr():
    assert isinstance(repr(Boolean), str)
    assert repr(Boolean) == 'Boolean'
    assert isinstance(repr(Categorical), str)
    assert repr(Categorical) == 'Categorical'


def test_instantiated_type_str():
    assert str(Categorical()) == 'Categorical'
    assert str(Boolean()) == 'Boolean'


def test_ordinal_order_errors():
    with pytest.raises(TypeError, match='Order values must be specified in a list or tuple'):
        Ordinal(order='not_valid')

    with pytest.raises(ValueError, match='Order values cannot contain duplicates'):
        Ordinal(order=['a', 'b', 'b'])


def test_ordinal_init_with_order():
    order = ['bronze', 'silver', 'gold']
    ordinal_from_list = Ordinal(order=order)
    assert ordinal_from_list.order == order

    order = ('bronze', 'silver', 'gold')
    ordinal_from_tuple = Ordinal(order=order)
    assert ordinal_from_tuple.order == order


def test_get_valid_dtype(sample_series):
    valid_dtype = Categorical._get_valid_dtype(type(sample_series))
    if ks and isinstance(sample_series, ks.Series):
        assert valid_dtype == 'string'
    else:
        assert valid_dtype == 'category'

    valid_dtype = Boolean._get_valid_dtype(type(sample_series))
    assert valid_dtype == 'bool'


def test_latlong_transform(latlong_df):
    df_type = str(type(latlong_df))
    dask = 'dask' in df_type
    koalas = 'koalas' in df_type
    nan = float('nan')

    expected_data = {
        'tuple_ints': [(1.0, 2.0), (3.0, 4.0)],
        'tuple_strings': [(1.0, 2.0), (3.0, 4.0)],
        'string_tuple': [(1.0, 2.0), (3.0, 4.0)],
        'bracketless_string_tuple': [(1.0, 2.0), (3.0, 4.0)],
        'list_strings': [(1.0, 2.0), (3.0, 4.0)],
        'combo_tuple_types': [(1.0, 2.0), (3.0, 4.0)],
        'null_value': [nan, (3.0, 4.0)],
        'null_latitude': [(nan, 2.0), (3.0, 4.0)],
        'both_null': [nan, (3.0, 4.0)],
    }

    latlong = LatLong()
    for column in latlong_df:
        series = latlong_df[column]
        actual = latlong.transform(series)

        if dask:
            actual = actual.compute()
        elif koalas:
            actual = actual.to_pandas()

        actual = actual.apply(pd.Series)
        series = pd.Series(expected_data[column])
        expected = series.apply(pd.Series)
        assert actual.equals(expected)


def test_datetime_transform(datetimes):
    datetime = Datetime()
    for series in datetimes:
        assert str(series.dtype) == 'object'
        transform = datetime.transform(series)
        assert str(transform.dtype) == 'datetime64[ns]'


def test_ordinal_transform(sample_series):
    series_type = str(type(sample_series))
    dask = 'dask' in series_type
    koalas = 'koalas' in series_type

    if dask or koalas:
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not supported')

    ordinal_incomplete_order = Ordinal(order=['a', 'b'])
    error_msg = re.escape("Ordinal column sample_series contains values that are not "
                          "present in the order values provided: ['c']")

    with pytest.raises(ValueError, match=error_msg):
        ordinal_incomplete_order.transform(sample_series)
