import pytest

import woodwork as ww
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    FullName,
    Ordinal
)
from woodwork.type_sys.utils import get_logical_types, str_to_logical_type


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


def test_get_logical_types():
    all_types = ww.type_system.registered_types
    logical_types = get_logical_types()

    for logical_type in all_types:
        assert logical_types[logical_type.__name__] == logical_type
        assert logical_types[logical_type.type_string] == logical_type

    assert len(logical_types) == 2 * len(all_types)


def test_str_to_logical_type():
    all_types = ww.type_system.registered_types

    with pytest.raises(ValueError, match='String test is not a valid logical type'):
        str_to_logical_type('test')
    assert str_to_logical_type('test', raise_error=False) is None

    for logical_type in all_types:
        assert str_to_logical_type(logical_type.__name__) == logical_type
        assert str_to_logical_type(logical_type.type_string) == logical_type

    assert str_to_logical_type('bOoLeAn') == Boolean
    assert str_to_logical_type('full_NAME') == FullName
    assert str_to_logical_type('FullnamE') == FullName

    ymd = '%Y-%m-%d'
    datetime_with_format = str_to_logical_type('datetime', params={'datetime_format': ymd})
    assert datetime_with_format.__class__ == Datetime
    assert datetime_with_format.datetime_format == ymd
    assert datetime_with_format == Datetime(datetime_format=ymd)

    datetime_no_format = str_to_logical_type('datetime', params={'datetime_format': None})
    assert datetime_no_format.__class__ == Datetime
    assert datetime_no_format.datetime_format is None
    assert datetime_no_format == Datetime()

    # When parameters are supplied in a non-empty dictionary, the logical type gets instantiated
    assert str_to_logical_type('full_NAME', params={}) == FullName
    assert datetime_no_format != Datetime


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
