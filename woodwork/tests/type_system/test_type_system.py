import pandas as pd

from woodwork.type_system.inference_functions import (
    categorical_func,
    double_func,
    integer_func
)
from woodwork.type_system.logical_types import (
    Categorical,
    CountryCode,
    Double,
    Integer,
    Ordinal,
    SubRegionCode
)
from woodwork.type_system.type_system import TypeSystem


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


def test_get_parent(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    assert type_sys._get_parent(CountryCode) == Categorical


def test_get_depth(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    type_sys.add_type(Ordinal, parent=CountryCode)
    assert type_sys._get_depth(Categorical) == 0
    assert type_sys._get_depth(CountryCode) == 1
    assert type_sys._get_depth(Ordinal) == 2


def test_update_inference_function(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    assert type_sys.inference_functions[Double] is double_func
    type_sys.update_inference_function(Double, integer_func)
    assert type_sys.inference_functions[Double] is integer_func


def test_update_relationship_no_children(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    type_sys.update_relationship(CountryCode, Integer)
    assert len(type_sys.relationships) == 2
    assert (Integer, CountryCode) in type_sys.relationships
    assert type_sys._get_parent(CountryCode) == Integer


def test_update_relationship_with_children(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    type_sys.add_type(SubRegionCode, parent=CountryCode)
    type_sys.update_relationship(CountryCode, Integer)
    assert len(type_sys.relationships) == 3
    assert (Integer, CountryCode) in type_sys.relationships
    assert (CountryCode, SubRegionCode) in type_sys.relationships
    assert type_sys._get_children(CountryCode) == [SubRegionCode]


def test_inference_multiple_matches_same_depth(default_relationships):
    def always_true(series):
        return True
    inference_functions = {
        Categorical: always_true,
        Double: always_true,
        Integer: always_true,
        CountryCode: always_true,
    }
    type_sys = TypeSystem(inference_functions=inference_functions,
                          relationships=default_relationships)
    type_sys.update_inference_function(Integer, always_true)
    type_sys.update_inference_function(CountryCode, always_true)
    inferred_type = type_sys.infer_logical_type(pd.Series([1, 2, 3]))
    # Should match CountryCode - same depth as Integer, but CountryCode parent
    # (Categorical) is tried and found first
    assert inferred_type == CountryCode


def test_inference_multiple_matches_different_depths(default_relationships):
    def always_true(series):
        return True
    inference_functions = {
        Categorical: always_true,
        Double: always_true,
        Integer: always_true,
        CountryCode: always_true,
    }
    type_sys = TypeSystem(inference_functions=inference_functions,
                          relationships=default_relationships)
    type_sys.update_inference_function(Integer, always_true)
    type_sys.update_inference_function(CountryCode, always_true)
    type_sys.add_type(SubRegionCode, inference_function=always_true, parent=CountryCode)
    inferred_type = type_sys.infer_logical_type(pd.Series([1, 2, 3]))
    # Should match SubRegionCode as it is the deepest match
    assert inferred_type == SubRegionCode
