from woodwork.type_system.type_system import TypeSystem
from woodwork.type_system.logical_types import (
    Categorical,
    CountryCode,
    Double,
    Integer,
    Ordinal
)
from woodwork.type_system.inference_functions import (
    categorical_func,
    double_func,
    integer_func
)


def test_type_system_init(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    assert len(type_sys.inference_functions) == 4
    assert type_sys.inference_functions[Double] is double_func
    assert type_sys.inference_functions[Integer] is integer_func
    assert type_sys.inference_functions[Categorical] is categorical_func
    assert len(type_sys.relationships) == 2
    assert type_sys.relationships[0] == (Double, Integer)
    assert type_sys.relationships[1] == (Categorical, CountryCode)


def test_type_system_registered_types(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    assert isinstance(type_sys.registered_types, list)
    assert set(type_sys.registered_types) == {Double, Integer, Categorical, CountryCode}


def test_type_system_root_types(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    assert isinstance(type_sys.root_types, list)
    assert set(type_sys.root_types) == {Double, Categorical}


def test_add_type_without_parent():
    type_sys = TypeSystem()
    type_sys.add_type(Double, inference_function=double_func)
    assert Double in type_sys.inference_functions.keys()
    assert type_sys.inference_functions[Double] is double_func

    type_sys.add_type(Integer)
    assert Integer in type_sys.inference_functions.keys()
    assert type_sys.inference_functions[Integer] is None


def test_add_type_with_parent():
    type_sys = TypeSystem()
    type_sys.add_type(Double, inference_function=double_func)
    type_sys.add_type(Integer, inference_function=integer_func, parent=Double)
    assert Integer in type_sys.inference_functions.keys()
    assert type_sys.inference_functions[Integer] is integer_func
    assert len(type_sys.relationships) == 1
    assert type_sys.relationships[0] == (Double, Integer)


def test_remove_type_no_children(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    type_sys.remove_type(Integer)
    assert len(type_sys.inference_functions) == 3
    assert Integer not in type_sys.inference_functions.keys()
    assert len(type_sys.relationships) == 1


def test_remove_type_with_children(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    type_sys.remove_type(Double)
    assert len(type_sys.inference_functions) == 3
    assert Double not in type_sys.inference_functions.keys()
    assert len(type_sys.relationships) == 2
    assert (None, Integer) in type_sys.relationships
    assert Integer in type_sys.root_types


def test_get_children(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    type_sys.add_type(Ordinal, parent=Categorical)
    children = type_sys._get_children(Categorical)
    assert isinstance(children, list)
    assert len(children) == 2
    assert set(children) == {CountryCode, Ordinal}


def test_get_parent():
    assert False


def test_get_depth():
    assert False


def test_update_inference_function():
    assert False


def test_update_relationship():
    assert False


def test_inference_multiple_matches_same_depth():
    assert False


def test_inference_multiple_matches_different_depths():
    assert False
