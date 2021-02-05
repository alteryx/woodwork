import woodwork as ww
from woodwork.type_sys.utils import _get_ltype_class
from woodwork.utils import _convert_input_to_set

from woodwork.logical_types import Boolean, Datetime, Ordinal


def _get_column_dict(name,
                     logical_type,
                     semantic_tags=None,
                     use_standard_tags=True,
                     column_description=None,
                     column_metadata=None):
    """Creates a dictionary that contains the typing information for a Schema column
    Args:
        name (str): The name of the column.
        logical_type (str, LogicalType): The column's LogicalType.
        semantic_tags (str, list, set, optional): The semantic tag(s) specified for the column.
        use_standard_tags (boolean, optional): If True, will add standard semantic tags to the column based
                specified logical type. Defaults to True.
        column_description (str, optional): User description of the column.
        column_metadata (dict[str -> json serializable], optional): Extra metadata provided by the user.
    """
    _validate_logical_type(logical_type)
    _validate_description(column_description)

    if column_metadata is None:
        column_metadata = {}
    _validate_metadata(column_metadata)

    semantic_tags = _get_column_tags(semantic_tags, logical_type, use_standard_tags, name)

    return {
        'name': name,
        'dtype': logical_type.pandas_dtype,
        'logical_type': logical_type,
        'semantic_tags': semantic_tags,
        'description': column_description,
        'metadata': column_metadata
    }


def _validate_tags(semantic_tags):
    """Verify user has not supplied tags that cannot be set directly"""
    if 'index' in semantic_tags:
        raise ValueError("Cannot add 'index' tag directly. To set a column as the index, "
                         "use Schema.set_index() instead.")
    if 'time_index' in semantic_tags:
        raise ValueError("Cannot add 'time_index' tag directly. To set a column as the time index, "
                         "use Schema.set_time_index() instead.")


def _validate_logical_type(logical_type):
    ltype_class = _get_ltype_class(logical_type)

    if ltype_class not in ww.type_system.registered_types:
        raise TypeError(f'logical_type {logical_type} is not a registered LogicalType.')
    if ltype_class == Ordinal and not isinstance(logical_type, Ordinal):
        raise TypeError("Must use an Ordinal instance with order values defined")


def _validate_description(column_description):
    if column_description is not None and not isinstance(column_description, str):
        raise TypeError("Column description must be a string")


def _validate_metadata(column_metadata):
    if not isinstance(column_metadata, dict):
        raise TypeError("Column metadata must be a dictionary")


def _get_column_tags(semantic_tags, logical_type, use_standard_tags, name):
    semantic_tags = _convert_input_to_set(semantic_tags, error_language=f'semantic_tags for column {name}')
    _validate_tags(semantic_tags)

    if use_standard_tags:
        semantic_tags = semantic_tags.union(logical_type.standard_tags)

    return semantic_tags


def _is_col_numeric(col_dict):
    return 'numeric' in col_dict['logical_type'].standard_tags


def _is_col_categorical(col_dict):
    return 'category' in col_dict['logical_type'].standard_tags


def _is_col_datetime(col_dict):
    return _get_ltype_class(col_dict['logical_type']) == Datetime


def _is_col_boolean(col_dict):
    return _get_ltype_class(col_dict['logical_type']) == Boolean
