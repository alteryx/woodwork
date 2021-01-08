from datetime import datetime

import pandas as pd

import woodwork as ww
from woodwork.utils import import_or_none

ks = import_or_none('databricks.koalas')


def col_is_datetime(col, datetime_format=None):
    """Determine if a dataframe column contains datetime values or not. Returns True if column
    contains datetimes, False if not. Optionally specify the datetime format string for the column."""
    if ks and isinstance(col, ks.Series):
        col = col.to_pandas()

    if (col.dtype.name.find('datetime') > -1 or
            (len(col) and isinstance(col.head(1), datetime))):
        return True

    # if it can be cast to numeric, it's not a datetime
    dropped_na = col.dropna()
    try:
        pd.to_numeric(dropped_na, errors='raise')
    except (ValueError, TypeError):
        # finally, try to cast to datetime
        if col.dtype.name.find('str') > -1 or col.dtype.name.find('object') > -1:
            try:
                pd.to_datetime(dropped_na, errors='raise', format=datetime_format)
            except Exception:
                return False
            else:
                return True

    return False


def _is_numeric_series(series, logical_type):
    '''
    Determines whether a series supplied to the DataTable will be considered numeric
    for the purposes of determining if it can be a time_index.

    '''
    if ks and isinstance(series, ks.Series):
        series = series.to_pandas()

    # If column can't be made to be numeric, don't bother checking Logical Type
    try:
        pd.to_numeric(series, errors='raise')
    except (ValueError, TypeError):
        return False

    if logical_type is not None:
        if isinstance(logical_type, str):
            logical_type = ww.type_system.str_to_logical_type(logical_type)

        # Allow numeric columns to be interpreted as Datetimes - doesn't allow strings even if they could be numeric
        if _get_ltype_class(logical_type) == ww.logical_types.Datetime and pd.api.types.is_numeric_dtype(series):
            return True
    else:
        logical_type = ww.type_system.infer_logical_type(series)

    return 'numeric' in logical_type.standard_tags


def list_logical_types():
    """Returns a dataframe describing all of the available Logical Types.

    Args:
        None

    Returns:
        pd.DataFrame: A dataframe containing details on each LogicalType, including
        the corresponding physical type and any standard semantic tags.
    """
    ltypes_df = pd.DataFrame(
        [{'name': ltype.__name__,
          'type_string': ltype.type_string,
          'description': ltype.__doc__,
          'physical_type': ltype.pandas_dtype,
          'standard_tags': ltype.standard_tags,
          'is_default_type': ltype in ww.type_system._default_inference_functions,
          'is_registered': ltype in ww.type_system.registered_types,
          'parent_type': ww.type_system._get_parent(ltype)}
            for ltype in ww.logical_types.LogicalType.__subclasses__()]
    )
    return ltypes_df.sort_values('name').reset_index(drop=True)


def list_semantic_tags():
    """Returns a dataframe describing all of the common semantic tags.

    Args:
        None

    Returns:
        pd.DataFrame: A dataframe containing details on each Semantic Tag, including
        the corresponding logical type(s).
    """
    sem_tags = {}
    for ltype in ww.type_system.registered_types:
        for tag in ltype.standard_tags:
            if tag in sem_tags:
                sem_tags[tag].append(ltype)
            else:
                sem_tags[tag] = [ltype]
    tags_df = pd.DataFrame(
        [{'name': tag,
          'is_standard_tag': True,
          'valid_logical_types': sem_tags[tag]}
         for tag in sem_tags]
    )
    tags_df = tags_df.append(
        pd.DataFrame([['index', False, [ww.type_system.str_to_logical_type(tag) for tag in ['integer', 'double', 'categorical', 'datetime']]],
                      ['time_index', False, [ww.type_system.str_to_logical_type('datetime')]],
                      ['date_of_birth', False, [ww.type_system.str_to_logical_type('datetime')]]
                      ], columns=tags_df.columns), ignore_index=True)
    return tags_df


def _get_ltype_class(ltype):
    if ltype in ww.type_system.registered_types:
        return ltype
    return ltype.__class__


def _get_specified_ltype_params(ltype):
    '''
    Gets a dictionary of a LogicalType's parameters.

    Note: If the logical type has not been instantiated, no parameters have
    been specified for the LogicalType, so no parameters will be returned
    even if that LogicalType can have parameters set.

    Args:
        ltype (LogicalType): An instantiated or uninstantiated LogicalType

    Returns:
        dict: The LogicalType's specified parameters.
    '''
    if ltype in ww.type_system.registered_types:
        # Do not reveal parameters for an uninstantiated LogicalType
        return {}
    return ltype.__dict__
