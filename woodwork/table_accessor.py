import inspect

import pandas as pd

from woodwork.schema import Schema


@pd.api.extensions.register_dataframe_accessor('ww')
class DataTableAccessor:
    def __init__(self, dataframe):
        self._dataframe = dataframe
        self._schema = None

    def init(self, make_index=False, already_sorted=False, **kwargs):
        # confirm all kwargs are present in the schema class - kwargs should be all the arguments from the Schema class
        _validate_schema_params(kwargs)

        # validate params as present - move param validation from Schema and add Accessor-specific ones

        #  Make index column if necessary

        # Type Inference for each column (parse ltype), updating dataframe (update col dtype)

        # make schema
        column_names = list(self._dataframe.columns)
        logical_types = {col_name: 'NaturalLanguage' for col_name in column_names}
        self._schema = Schema(column_names=column_names, logical_types=logical_types, **kwargs)

        # sort columns based on index


def _validate_schema_params(schema_params_dict):
    possible_schema_params = inspect.signature(Schema).parameters
    for param in schema_params_dict.keys():
        if param not in possible_schema_params:
            raise TypeError(f'Parameter {param} does not exist on the Schema class.')
