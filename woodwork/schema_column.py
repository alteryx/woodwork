from woodwork.utils import _convert_input_to_set, _parse_column_logical_type


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
        semantic_tags (str, list, set): 

    """
    _validate_description(column_description)

    column_metadata = _validate_metadata(column_metadata)

    logical_type = _parse_column_logical_type(logical_type, name)

    semantic_tags = _get_column_tags(semantic_tags, logical_type, use_standard_tags, name)

    return {
        'name': name,
        'dtype': logical_type.pandas_dtype,  # --> should either be pandas dtype or backup depending on if it's a pandas schema
        'logical_type': logical_type,
        'semantic_tags': semantic_tags,
        'use_standard_tags': use_standard_tags,
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


def _validate_description(column_description):
    if column_description is not None and not isinstance(column_description, str):
        raise TypeError("Column description must be a string")


def _validate_metadata(column_metadata):
    if column_metadata is None:
        column_metadata = {}
    if not isinstance(column_metadata, dict):
        raise TypeError("Column metadata must be a dictionary")
    return column_metadata


def _get_column_tags(semantic_tags, logical_type, use_standard_tags, name):
    semantic_tags = _convert_input_to_set(semantic_tags, error_language=f'semantic_tags for column {name}')
    _validate_tags(semantic_tags)

    if use_standard_tags:
        semantic_tags = semantic_tags.union(logical_type.standard_tags)

    return semantic_tags
