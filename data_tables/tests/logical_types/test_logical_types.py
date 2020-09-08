from data_tables.logical_types import Boolean, Categorical


def test_logical_eq():
    assert Boolean == Boolean
    assert Boolean() == Boolean()
    assert Categorical != Boolean


def test_logical_repr():
    assert isinstance(repr(Boolean), str)
    assert repr(Boolean) == 'Boolean'
    assert isinstance(repr(Categorical), str)
    assert repr(Categorical) == 'Categorical'
