import pandas as pd
import pytest

import woodwork as ww
from woodwork.logical_types import (
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    FullName,
    Integer,
    LogicalType,
    NaturalLanguage,
    Ordinal,
    SubRegionCode
)
from woodwork.type_sys.inference_functions import (
    categorical_func,
    double_func,
    integer_func
)
from woodwork.type_sys.type_system import TypeSystem


def test_type_system_init(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships)
    assert len(type_sys.inference_functions) == 5
    assert type_sys.inference_functions[Double] is double_func
    assert type_sys.inference_functions[Integer] is integer_func
    assert type_sys.inference_functions[Categorical] is categorical_func
    assert len(type_sys.relationships) == 2
    assert type_sys.relationships[0] == (Double, Integer)
    assert type_sys.relationships[1] == (Categorical, CountryCode)


def test_add_type_validation_errors(type_sys):
    error_msg = 'logical_type must be a valid LogicalType'
    with pytest.raises(TypeError, match=error_msg):
        type_sys.add_type(logical_type=1)

    with pytest.raises(TypeError, match=error_msg):
        type_sys.add_type(logical_type=Double())

    error_msg = 'inference_function must be a function'
    with pytest.raises(TypeError, match=error_msg):
        type_sys.add_type(logical_type=Ordinal, inference_function='not a function')

    error_msg = 'parent must be a valid LogicalType'
    with pytest.raises(ValueError, match=error_msg):
        type_sys.add_type(logical_type=Ordinal, parent=1)

    with pytest.raises(ValueError, match=error_msg):
        type_sys.add_type(logical_type=Ordinal, parent=Double())


def test_remove_type_validation_errors(type_sys):
    error_msg = 'logical_type must be a valid LogicalType'
    with pytest.raises(TypeError, match=error_msg):
        type_sys.add_type(logical_type=1)


def test_update_inference_function_validation_errors(type_sys):
    error_msg = 'logical_type must be a valid LogicalType'
    with pytest.raises(TypeError, match=error_msg):
        type_sys.update_inference_function(logical_type=1, inference_function=None)

    error_msg = 'inference_function must be a function'
    with pytest.raises(TypeError, match=error_msg):
        type_sys.update_inference_function(logical_type=Ordinal, inference_function='not a function')


def test_update_relationship_validation_errors(type_sys):
    error_msg = 'logical_type must be a valid LogicalType'
    with pytest.raises(TypeError, match=error_msg):
        type_sys.update_relationship(logical_type=1, parent=Ordinal)

    error_msg = 'parent must be a valid LogicalType'
    with pytest.raises(ValueError, match=error_msg):
        type_sys.update_relationship(logical_type=Ordinal, parent=1)


def test_type_system_default_type(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships,
                          default_type=SubRegionCode)
    assert type_sys.default_type == SubRegionCode
    type_sys.update_inference_function(Categorical, None)
    test_series = pd.Series(['a', 'b', 'c'])
    assert type_sys.infer_logical_type(test_series) == SubRegionCode
    assert SubRegionCode in type_sys.registered_types


def test_type_system_default_type_remove_error(default_inference_functions, default_relationships):
    type_sys = TypeSystem(inference_functions=default_inference_functions,
                          relationships=default_relationships,
                          default_type=SubRegionCode)
    error_msg = "Default LogicalType cannot be removed"
    with pytest.raises(ValueError, match=error_msg):
        type_sys.remove_type(SubRegionCode)
    with pytest.raises(ValueError, match=error_msg):
        type_sys.remove_type('SubRegionCode')
    with pytest.raises(ValueError, match=error_msg):
        type_sys.remove_type('sub_region_code')


def test_type_system_registered_types(type_sys):
    assert isinstance(type_sys.registered_types, list)
    assert set(type_sys.registered_types) == {Double, Integer, Categorical, CountryCode, NaturalLanguage}


def test_type_system_root_types(type_sys):
    assert isinstance(type_sys.root_types, list)
    assert set(type_sys.root_types) == {Double, Categorical, NaturalLanguage}


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


def test_add_duplicate_ltype(type_sys):
    inference_fn = type_sys.inference_functions[ww.logical_types.Integer]

    assert ww.type_system.str_to_logical_type('Integer') == ww.logical_types.Integer

    class Integer(LogicalType):
        pandas_dtype = 'string'

    error_msg = 'Logical Type with name Integer already present in the Type System. Please rename the LogicalType or remove existing one.'
    with pytest.raises(ValueError, match=error_msg):
        type_sys.add_type(Integer, inference_function=inference_fn)

    type_sys.remove_type(ww.logical_types.Integer)
    type_sys.add_type(Integer, inference_function=inference_fn)

    ltype = type_sys.str_to_logical_type('Integer')
    assert ltype.pandas_dtype == 'string'
    assert ltype == Integer

    type_sys.reset_defaults()
    ltype = type_sys.str_to_logical_type('Integer')
    assert ltype.pandas_dtype == 'Int64'
    assert ltype == ww.logical_types.Integer


def test_remove_type_no_children(type_sys):
    type_sys.remove_type(Integer)
    assert len(type_sys.inference_functions) == 4
    assert Integer not in type_sys.inference_functions.keys()
    assert len(type_sys.relationships) == 1
    type_sys.remove_type('CountryCode')
    assert len(type_sys.inference_functions) == 3
    assert CountryCode not in type_sys.inference_functions.keys()
    assert len(type_sys.relationships) == 0


def test_remove_type_with_children(type_sys):
    type_sys.remove_type(Double)
    assert len(type_sys.inference_functions) == 4
    assert Double not in type_sys.inference_functions.keys()
    assert len(type_sys.relationships) == 2
    assert (None, Integer) in type_sys.relationships
    assert Integer in type_sys.root_types


def test_get_children(type_sys):
    type_sys.add_type(Ordinal, parent=Categorical)
    children = type_sys._get_children(Categorical)
    assert isinstance(children, list)
    assert len(children) == 2
    assert set(children) == {CountryCode, Ordinal}


def test_get_parent(type_sys):
    assert type_sys._get_parent(CountryCode) == Categorical
    assert type_sys._get_parent(Integer) == Double


def test_get_depth(type_sys):
    type_sys.add_type(Ordinal, parent=CountryCode)
    assert type_sys._get_depth(Categorical) == 0
    assert type_sys._get_depth(CountryCode) == 1
    assert type_sys._get_depth(Ordinal) == 2


def test_update_inference_function(type_sys):
    assert type_sys.inference_functions[Double] is double_func
    type_sys.update_inference_function(Double, integer_func)
    assert type_sys.inference_functions[Double] is integer_func
    type_sys.update_inference_function('CountryCode', integer_func)
    assert type_sys.inference_functions[CountryCode] is integer_func


def test_update_relationship_no_children(type_sys):
    type_sys.update_relationship(CountryCode, Integer)
    assert len(type_sys.relationships) == 2
    assert (Integer, CountryCode) in type_sys.relationships
    assert type_sys._get_parent(CountryCode) == Integer


def test_update_relationship_string_input(type_sys):
    type_sys.update_relationship('CountryCode', 'Integer')
    assert len(type_sys.relationships) == 2
    assert (Integer, CountryCode) in type_sys.relationships
    assert type_sys._get_parent(CountryCode) == Integer


def test_update_relationship_with_children(type_sys):
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
                          relationships=default_relationships,
                          default_type=NaturalLanguage)
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
                          relationships=default_relationships,
                          default_type=NaturalLanguage)
    type_sys.update_inference_function(Integer, always_true)
    type_sys.update_inference_function(CountryCode, always_true)
    type_sys.add_type(SubRegionCode, inference_function=always_true, parent=CountryCode)
    inferred_type = type_sys.infer_logical_type(pd.Series([1, 2, 3]))
    # Should match SubRegionCode as it is the deepest match
    assert inferred_type == SubRegionCode


def test_reset_defaults(type_sys, default_inference_functions, default_relationships):
    type_sys.update_inference_function('Integer', None)
    type_sys.update_relationship(CountryCode, parent=NaturalLanguage)
    type_sys.default_type = Categorical
    type_sys.reset_defaults()

    assert type_sys.inference_functions == default_inference_functions
    assert type_sys.relationships == default_relationships
    assert type_sys.default_type == NaturalLanguage


def test_get_logical_types():
    all_types = ww.type_system.registered_types
    logical_types = ww.type_system._get_logical_types()

    for logical_type in all_types:
        assert logical_types[logical_type.__name__] == logical_type
        assert logical_types[logical_type.type_string] == logical_type

    assert len(logical_types) == 2 * len(all_types)


def test_str_to_logical_type():
    all_types = ww.type_system.registered_types

    with pytest.raises(ValueError, match='String test is not a valid logical type'):
        ww.type_system.str_to_logical_type('test')
    assert ww.type_system.str_to_logical_type('test', raise_error=False) is None

    for logical_type in all_types:
        assert ww.type_system.str_to_logical_type(logical_type.__name__) == logical_type
        assert ww.type_system.str_to_logical_type(logical_type.type_string) == logical_type

    assert ww.type_system.str_to_logical_type('bOoLeAn') == Boolean
    assert ww.type_system.str_to_logical_type('full_NAME') == FullName
    assert ww.type_system.str_to_logical_type('FullnamE') == FullName

    ymd = '%Y-%m-%d'
    datetime_with_format = ww.type_system.str_to_logical_type('datetime', params={'datetime_format': ymd})
    assert datetime_with_format.__class__ == Datetime
    assert datetime_with_format.datetime_format == ymd
    assert datetime_with_format == Datetime(datetime_format=ymd)

    datetime_no_format = ww.type_system.str_to_logical_type('datetime', params={'datetime_format': None})
    assert datetime_no_format.__class__ == Datetime
    assert datetime_no_format.datetime_format is None
    assert datetime_no_format == Datetime()

    # When parameters are supplied in a non-empty dictionary, the logical type gets instantiated
    assert ww.type_system.str_to_logical_type('full_NAME', params={}) == FullName
    assert datetime_no_format != Datetime

    # Input a different type system
    new_type_sys = TypeSystem()
    with pytest.raises(ValueError, match='String Integer is not a valid logical type'):
        new_type_sys.str_to_logical_type('Integer')
    new_type_sys.add_type(Boolean)
    assert Boolean == new_type_sys.str_to_logical_type('Boolean')
