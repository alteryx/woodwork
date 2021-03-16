import pandas as pd

from woodwork.logical_types import Categorical, LogicalType


def test_register_custom_logical_type(type_sys):
    class CustomLogicalType(LogicalType):
        primary_dtype = 'string'

    def custom_func(series):
        return True

    type_sys.add_type(CustomLogicalType,
                      inference_function=custom_func,
                      parent='Categorical')
    assert CustomLogicalType in type_sys.registered_types
    assert (Categorical, CustomLogicalType) in type_sys.relationships
    assert type_sys.inference_functions[CustomLogicalType] is custom_func
    assert type_sys.infer_logical_type(pd.Series(['a', 'b', 'a'])) == CustomLogicalType
