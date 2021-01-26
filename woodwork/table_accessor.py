import inspect

import pandas as pd
import woodwork as ww

from woodwork.logical_types import Datetime, LatLong, Ordinal
from woodwork.schema import Schema
from woodwork.utils import import_or_none, _parse_column_logical_type
from woodwork.type_sys.utils import _get_ltype_class

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')
if ks:
    ks.set_option('compute.ops_on_diff_frames', True)


@pd.api.extensions.register_dataframe_accessor('ww')
class WoodworkTableAccessor:
    def __init__(self, dataframe):
        self._dataframe = dataframe
        self._schema = None

    def init(self, index=None, time_index=None, logical_types=None, make_index=False, already_sorted=False, **kwargs):
        # confirm all kwargs are present in the schema class - kwargs should be all the arguments from the Schema class
        _validate_schema_params(kwargs)
        _validate_accessor_params(self._dataframe, index, make_index, time_index, logical_types)

        if make_index:
            _make_index(self._dataframe, index)

        # Perform type inference and update underlying data
        parsed_logical_types = {}
        for name in self._dataframe.columns:
            series = self._dataframe[name]

            logical_type = None
            if logical_types:
                logical_type = logical_types.get(name)

            logical_type = _parse_column_logical_type(series, logical_type, name)
            parsed_logical_types[name] = logical_type

            updated_series = _update_column_dtype(series, logical_type, name)
            self._dataframe[name] = updated_series

        # Create the Schema
        column_names = list(self._dataframe.columns)
        self._schema = Schema(column_names=column_names,
                              logical_types=parsed_logical_types,
                              index=index,  # --> do a test that this doesnt double up weirdly
                              time_index=time_index, **kwargs)

        # sort columns based on index

    def __getattribute__(self, attr):
        '''
            If method is present on the Accessor, uses that method. 
            If the method is present on Schema, uses that method. 
        '''
        try:
            # --> see if there's a way to use hasattr on the object in its own
            return object.__getattribute__(self, attr)
        except AttributeError:
            schema = object.__getattribute__(self, '_schema')
            if hasattr(schema, attr):
                schema_attr = getattr(schema, attr)

                # logical_type attr can return an uninstantiated LogicalType which we don't want to interpret as callable
                if callable(schema_attr) and attr != 'logical_type':
                    def wrapper(*args, **kwargs):
                        return schema_attr(*args, **kwargs)
                    return wrapper
                else:
                    return schema_attr

    @property
    def schema(self):
        return self._schema


def _validate_schema_params(schema_params_dict):
    possible_schema_params = inspect.signature(Schema).parameters
    for param in schema_params_dict.keys():
        if param not in possible_schema_params:
            raise TypeError(f'Parameter {param} does not exist on the Schema class.')


def _validate_accessor_params(dataframe, index, make_index, time_index, logical_types):
    # --> figure out best way to utilize Schema checks code without repeating checks!!!!
    #  --> either remove redundant checks or maybe pass 'already checked' param??
    #  There has to be a balance betwen where we check each param - want users to b e able to make Schemas directly without creating errors
    # --> maybe turn repetetive cheks into assertions
    _check_unique_column_names(dataframe)
    if index is not None or make_index:
        _check_index(dataframe, index, make_index)
    if logical_types:
        _check_logical_types(dataframe.columns, logical_types)
    if time_index is not None:
        datetime_format = None
        logical_type = None
        if logical_types is not None and time_index in logical_types:
            logical_type = logical_types[time_index]
            if _get_ltype_class(logical_types[time_index]) == Datetime:
                datetime_format = logical_types[time_index].datetime_format

        _check_time_index(dataframe, time_index, datetime_format=datetime_format, logical_type=logical_type)


def _check_unique_column_names(dataframe):
    if not dataframe.columns.is_unique:
        raise IndexError('Dataframe cannot contain duplicate columns names')


def _check_index(dataframe, index, make_index=False):
    # --> definitely might be able to reuse
    if not make_index and index not in dataframe.columns:
        # User specifies an index that is not in the dataframe, without setting make_index to True
        raise LookupError(f'Specified index column `{index}` not found in dataframe. To create a new index column, set make_index to True.')
    if index is not None and not make_index and isinstance(dataframe, pd.DataFrame) and not dataframe[index].is_unique:
        # User specifies an index that is in the dataframe but not unique
        # Does not check for Dask as Dask does not support is_unique
        raise IndexError('Index column must be unique')
    if make_index and index is not None and index in dataframe.columns:
        # User sets make_index to True, but supplies an index name that matches a column already present
        raise IndexError('When setting make_index to True, the name specified for index cannot match an existing column name')
    if make_index and index is None:
        # User sets make_index to True, but does not supply a name for the index
        raise IndexError('When setting make_index to True, the name for the new index must be specified in the index parameter')


def _check_time_index(dataframe, time_index, datetime_format=None, logical_type=None):
    if time_index not in dataframe.columns:
        raise LookupError(f'Specified time index column `{time_index}` not found in dataframe')
    if not (_is_numeric_series(dataframe[time_index], logical_type) or
            col_is_datetime(dataframe[time_index], datetime_format=datetime_format)):
        raise TypeError('Time index column must contain datetime or numeric values')


def _check_logical_types(dataframe_columns, logical_types):
    # --> definitely reusable but maybe not neccessary both places???
    if not isinstance(logical_types, dict):
        raise TypeError('logical_types must be a dictionary')
    cols_not_found = set(logical_types.keys()).difference(set(dataframe_columns))
    if cols_not_found:
        raise LookupError('logical_types contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')


def _make_index(dataframe, index):
    if dd and isinstance(dataframe, dd.DataFrame):
        dataframe[index] = 1
        dataframe[index] = dataframe[index].cumsum() - 1
    elif ks and isinstance(dataframe, ks.DataFrame):
        raise TypeError('Cannot make index on a Koalas DataFrame.')
    else:
        dataframe.insert(0, index, range(len(dataframe)))


def _update_column_dtype(series, logical_type, name):
    """Update the dtype of the underlying series to match the dtype corresponding
    to the LogicalType for the column."""
    if isinstance(logical_type, Ordinal):
        logical_type._validate_data(series)
    elif _get_ltype_class(logical_type) == LatLong:
        # Reformat LatLong columns to be a length two tuple (or list for Koalas) of floats
        if dd and isinstance(series, dd.Series):
            name = series.name
            meta = (series, tuple([float, float]))
            series = series.apply(_reformat_to_latlong, meta=meta)
            series.name = name
        elif ks and isinstance(series, ks.Series):
            formattedseries = series.to_pandas().apply(_reformat_to_latlong, use_list=True)
            series = ks.from_pandas(formattedseries)
        else:
            series = series.apply(_reformat_to_latlong)

    if logical_type.pandas_dtype != str(series.dtype):
        # Update the underlying series
        try:
            if _get_ltype_class(logical_type) == Datetime:
                if dd and isinstance(series, dd.Series):
                    name = series.name
                    series = dd.to_datetime(series, format=logical_type.datetime_format)
                    series.name = name
                elif ks and isinstance(series, ks.Series):
                    series = ks.Series(ks.to_datetime(series.to_numpy(),
                                                      format=logical_type.datetime_format),
                                       name=series.name)
                else:
                    series = pd.to_datetime(series, format=logical_type.datetime_format)
            else:
                if ks and isinstance(series, ks.Series) and logical_type.backup_dtype:
                    new_dtype = logical_type.backup_dtype
                else:
                    new_dtype = logical_type.pandas_dtype
                series = series.astype(new_dtype)
        except (TypeError, ValueError):
            error_msg = f'Error converting datatype for column {name} from type {str(series.dtype)} ' \
                f'to type {logical_type.pandas_dtype}. Please confirm the underlying data is consistent with ' \
                f'logical type {logical_type}.'
            raise TypeError(error_msg)
    return series
