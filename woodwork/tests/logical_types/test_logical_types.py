import pytest

from woodwork.logical_types import (
    Boolean,
    Categorical,
    FullName,
    LogicalType,
    Ordinal,
    get_logical_types,
    str_to_logical_type
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


def test_instantiated_type_str():
    assert str(Categorical()) == 'Categorical'
    assert str(Boolean()) == 'Boolean'


def test_get_logical_types():
    all_types = LogicalType.__subclasses__()
    logical_types = get_logical_types()

    for logical_type in all_types:
        assert logical_types[logical_type.__name__] == logical_type
        assert logical_types[logical_type.type_string] == logical_type

    assert len(logical_types) == 2 * len(all_types)


def test_str_to_logical_type():
    all_types = LogicalType.__subclasses__()

    with pytest.raises(ValueError, match='String test is not a valid logical type'):
        str_to_logical_type('test')
    assert str_to_logical_type('test', raise_error=False) is None

    for logical_type in all_types:
        assert str_to_logical_type(logical_type.__name__) == logical_type
        assert str_to_logical_type(logical_type.type_string) == logical_type

    assert str_to_logical_type('bOoLeAn') == Boolean
    assert str_to_logical_type('full_NAME') == FullName
    assert str_to_logical_type('FullnamE') == FullName


def test_ordinal_ranking_errors():
    with pytest.raises(TypeError, match='Ordinal ranking values must be specified in a list or tuple'):
        Ordinal(ranking='not_valid')

    with pytest.raises(ValueError, match='Ordinal ranking values cannot contain duplicates'):
        Ordinal(ranking=['a', 'b', 'b'])


def test_ordinal_init_with_ranking():
    ranks = ['bronze', 'silver', 'gold']
    ordinal_from_list = Ordinal(ranking=ranks)
    assert ordinal_from_list.ranking == ranks

    ranks = ('bronze', 'silver', 'gold')
    ordinal_from_tuple = Ordinal(ranking=ranks)
    assert ordinal_from_tuple.ranking == ranks
