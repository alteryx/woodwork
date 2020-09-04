from data_tables.logical_types import (
    Boolean,
    Categorical
)


def test_logical_eq():
    assert Boolean().__eq__(Boolean())
    assert not Categorical().__eq__(Boolean())


def test_logical_repr():
    assert isinstance(Boolean().__repr__(), str)
    assert isinstance(Categorical().__repr__(), str)