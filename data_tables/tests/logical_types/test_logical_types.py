from data_tables.logical_types import (
    Boolean,
    Categorical,
    LogicalType,
    get_logical_types
)


def test_logical_eq():
    assert Boolean == Boolean
    assert Boolean() == Boolean()
    assert Categorical != Boolean


def test_logical_repr():
    assert isinstance(repr(Boolean), str)
    assert repr(Boolean) == 'Boolean'
    assert isinstance(repr(Categorical), str)
    assert repr(Categorical) == 'Categorical'


def test_get_logical_types():
    all_types = LogicalType.__subclasses__()
    logical_types = get_logical_types()

    for logical_type in all_types:
        assert logical_types[logical_type.__name__] == logical_type
        assert logical_types[logical_type.type_string] == logical_type

    assert len(logical_types) == 2 * len(all_types)
