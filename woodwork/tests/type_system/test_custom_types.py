import pandas as pd
import pandas.api.types as pdtypes

import woodwork as ww
from woodwork.logical_types import Categorical, Double, LogicalType


def test_register_custom_logical_type(type_sys):
    class CustomLogicalType(LogicalType):
        pandas_dtype = 'string'

    def custom_func(series):
        return True

    type_sys.add_type(CustomLogicalType,
                      inference_function=custom_func,
                      parent='Categorical')
    assert CustomLogicalType in type_sys.registered_types
    assert (Categorical, CustomLogicalType) in type_sys.relationships
    assert type_sys.inference_functions[CustomLogicalType] is custom_func
    assert type_sys.infer_logical_type(pd.Series(['a', 'b', 'a'])) == CustomLogicalType


def test_custom_type_with_datatable(sample_df):
    class AgesAbove20(LogicalType):
        pandas_dtype = 'float64'
        standard_tags = {'age', 'numeric'}

    def ages_func(series):
        if all(series > 20):
            return True
        return False

    ww.type_system.add_type(AgesAbove20, inference_function=ages_func, parent='Integer')
    dt = ww.DataTable(sample_df)
    assert dt['age'].logical_type == AgesAbove20
    assert dt['age'].semantic_tags == {'age', 'numeric'}
    assert dt.to_dataframe()['age'].dtype == 'float64'
    # Reset global type system to original settings
    ww.type_system.reset_defaults()


def test_override_default_function(sample_df):
    def new_double_func(series):
        if pdtypes.is_integer_dtype(series.dtype):
            return True
        return False

    # Update functions to cause 'age' to be recognized as Double instead fo Integer
    ww.type_system.update_inference_function('Double', inference_function=new_double_func)
    ww.type_system.update_inference_function('Integer', inference_function=None)
    dt = ww.DataTable(sample_df)
    assert dt['age'].logical_type == Double
    assert dt.to_dataframe()['age'].dtype == 'float64'
    # Reset global type system to original settings
    ww.type_system.reset_defaults()
