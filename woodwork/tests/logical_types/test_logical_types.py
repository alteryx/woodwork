import pytest

from woodwork.logical_types import (
    Address,
    Age,
    AgeNullable,
    Boolean,
    Categorical,
    Datetime,
    Ordinal
)

COLUMN_TO_LOGICAL_TYPE = {
    'argnames': 'column,logical_type',
    'argvalues': [
        ('email', Address),
        ('id', Age),
        ('age', AgeNullable),
    ],
}


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


@pytest.mark.parametrize(**COLUMN_TO_LOGICAL_TYPE)
def test_init(sample_df, column, logical_type):
    sample_df.ww.init(logical_types={column: logical_type})
    actual = sample_df.ww[column].ww.logical_type
    info = f'"{column}" not initialized as "{logical_type}"'
    assert actual == logical_type, info


@pytest.mark.parametrize(**COLUMN_TO_LOGICAL_TYPE)
def test_set_types(sample_df, column, logical_type):
    sample_df.ww.init()
    before = sample_df.ww[column].ww.logical_type
    info = f'"{column}" already set as "{logical_type}""'
    assert before != logical_type, info

    sample_df.ww.set_types(logical_types={column: logical_type})
    after = sample_df.ww[column].ww.logical_type
    info = f'"{column}" not set as "{logical_type}"'
    assert after == logical_type, info
