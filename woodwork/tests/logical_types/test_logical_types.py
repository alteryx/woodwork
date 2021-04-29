import pytest

from woodwork.logical_types import Boolean, Categorical, Datetime, Ordinal


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
