import importlib
import re
from datetime import datetime

import pandas as pd

import woodwork as ww


def import_or_none(library):
    '''
    Attemps to import the requested library.

    Args:
        library (str): the name of the library
    Returns: the library if it is installed, else None
    '''
    try:
        return importlib.import_module(library)
    except ImportError:
        return None


ks = import_or_none('databricks.koalas')


def camel_to_snake(s):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def _convert_input_to_set(semantic_tags, error_language='semantic_tags'):
    """Takes input as a single string, a list of strings, or a set of strings
        and returns a set with the supplied values. If no values are supplied,
        an empty set will be returned."""
    if not semantic_tags:
        return set()

    if type(semantic_tags) not in [list, set, str]:
        raise TypeError(f"{error_language} must be a string, set or list")

    if isinstance(semantic_tags, str):
        return {semantic_tags}

    if isinstance(semantic_tags, list):
        semantic_tags = set(semantic_tags)

    if not all([isinstance(tag, str) for tag in semantic_tags]):
        raise TypeError(f"{error_language} must contain only strings")

    return semantic_tags


def col_is_datetime(col, datetime_format=None):
    """Determine if a dataframe column contains datetime values or not. Returns True if column
    contains datetimes, False if not. Optionally specify the datetime format string for the column."""
    if ks and isinstance(col, ks.Series):
        col = col.to_pandas()

    if (col.dtype.name.find('datetime') > -1 or
            (len(col) and isinstance(col.head(1), datetime))):
        return True

    # if it can be casted to numeric, it's not a datetime
    dropped_na = col.mask(col.eq('None')).dropna()
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
            logical_type = ww.logical_types.str_to_logical_type(logical_type)

        # Allow numeric columns to be interpreted as Datetimes - doesn't allow strings even if they could be numeric
        if _get_ltype_class(logical_type) == ww.logical_types.Datetime and pd.api.types.is_numeric_dtype(series):
            return True
    else:
        logical_type = ww.datacolumn.infer_logical_type(series)

    return 'numeric' in logical_type.standard_tags


def list_logical_types():
    """Returns a dataframe describing all of the available Logical Types.

    Args:
        None

    Returns:
        pd.DataFrame: A dataframe containing details on each LogicalType, including
        the corresponding physical type and any standard semantic tags.
    """
    return pd.DataFrame(
        [{'name': ltype.__name__,
          'type_string': ltype.type_string,
          'description': ltype.__doc__,
          'physical_type': ltype.pandas_dtype,
          'standard_tags': ltype.standard_tags}
            for ltype in ww.logical_types.LogicalType.__subclasses__()]
    )


def list_semantic_tags():
    """Returns a dataframe describing all of the common semantic tags.

    Args:
        None

    Returns:
        pd.DataFrame: A dataframe containing details on each Semantic Tag, including
        the corresponding logical type(s).
    """
    sem_tags = {}
    for ltype in ww.logical_types.LogicalType.__subclasses__():
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
        pd.DataFrame([['index', False, [ww.logical_types.str_to_logical_type(tag) for tag in ['integer', 'double', 'categorical', 'datetime']]],
                      ['time_index', False, [ww.logical_types.str_to_logical_type('datetime')]],
                      ['date_of_birth', False, [ww.logical_types.str_to_logical_type('datetime')]]
                      ], columns=tags_df.columns), ignore_index=True)
    return tags_df


def _get_mode(series):
    """Get the mode value for a series"""
    mode_values = series.mode()
    if len(mode_values) > 0:
        return mode_values[0]
    return None


def read_csv(filepath=None,
             name=None,
             index=None,
             time_index=None,
             semantic_tags=None,
             logical_types=None,
             use_standard_tags=True,
             **kwargs):
    """Read data from the specified CSV file and return a Woodwork DataTable

    Args:
        filepath (str): A valid string path to the file to read
        name (str, optional): Name used to identify the datatable.
        index (str, optional): Name of the index column in the dataframe.
        time_index (str, optional): Name of the time index column in the dataframe.
        semantic_tags (dict, optional): Dictionary mapping column names in the dataframe to the
            semantic tags for the column. The keys in the dictionary should be strings
            that correspond to columns in the underlying dataframe. There are two options for
            specifying the dictionary values:
            (str): If only one semantic tag is being set, a single string can be used as a value.
            (list[str] or set[str]): If multiple tags are being set, a list or set of strings can be
            used as the value.
            Semantic tags will be set to an empty set for any column not included in the
            dictionary.
        logical_types (dict[str -> LogicalType], optional): Dictionary mapping column names in
            the dataframe to the LogicalType for the column. LogicalTypes will be inferred
            for any columns not present in the dictionary.
        use_standard_tags (bool, optional): If True, will add standard semantic tags to columns based
            on the inferred or specified logical type for the column. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the underlying ``pandas.read_csv`` function. For more
            information on available keywords refer to the pandas documentation.

    Returns:
        woodwork.DataTable: DataTable created from the specified CSV file
    """
    dataframe = pd.read_csv(filepath, **kwargs)
    return ww.DataTable(dataframe,
                        name=name,
                        index=index,
                        time_index=time_index,
                        semantic_tags=semantic_tags,
                        logical_types=logical_types,
                        use_standard_tags=use_standard_tags)


def _get_ltype_class(ltype):
    if ltype in ww.logical_types.LogicalType.__subclasses__():
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
    if ltype in ww.logical_types.LogicalType.__subclasses__():
        # Do not reveal parameters for an uninstantiated LogicalType
        return {}
    return ltype.__dict__


def _new_dt_including(datatable, new_data):
    '''
    Creates a new DataTable with specified data and columns

    Args:
        datatable (DataTable): DataTable with desired information

        new_data (DataFrame): subset of original DataTable
    Returns:
        DataTable: New DataTable with attributes from original DataTable but data from new DataTable
    '''
    cols = new_data.columns
    new_semantic_tags = {col_name: semantic_tag_set for col_name, semantic_tag_set
                         in datatable.semantic_tags.items() if col_name in cols}
    new_logical_types = {col_name: logical_type for col_name, logical_type
                         in datatable.logical_types.items() if col_name in cols}
    new_index = datatable.index if datatable.index in cols else None
    new_time_index = datatable.time_index if datatable.time_index in cols else None
    if new_index:
        new_semantic_tags[new_index] = new_semantic_tags[new_index].difference({'index'})
    if new_time_index:
        new_semantic_tags[new_time_index] = new_semantic_tags[new_time_index].difference({'time_index'})
    return ww.DataTable(new_data,
                        name=datatable.name,
                        index=new_index,
                        time_index=new_time_index,
                        semantic_tags=new_semantic_tags,
                        logical_types=new_logical_types,
                        use_standard_tags=datatable.use_standard_tags)


def import_or_raise(library, error_msg):
    '''
    Attempts to import the requested library.  If the import fails, raises an
    ImportError with the supplied error message.

    Args:
        library (str): the name of the library
        error_msg (str): error message to return if the import fails
    '''
    try:
        return importlib.import_module(library)
    except ImportError:
        raise ImportError(error_msg)


def _is_s3(string):
    '''
    Checks if the given string is a s3 path.
    Returns a boolean.
    '''
    return "s3://" in string


def _is_url(string):
    '''
    Checks if the given string is an url path.
    Returns a boolean.
    '''
    return 'http' in string
