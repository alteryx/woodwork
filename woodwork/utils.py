import re
from datetime import datetime

import pandas as pd


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
    if (col.dtype.name.find('datetime') > -1 or
            (len(col) and isinstance(col.iloc[0], datetime))):
        return True

    # if it can be casted to numeric, it's not a datetime
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
