import warnings

import woodwork as ww
from woodwork.exceptions import (
    DuplicateTagsWarning,
    StandardTagsChangedWarning
)
from woodwork.logical_types import Boolean, Datetime, Ordinal
from woodwork.type_sys.utils import _get_ltype_class
from woodwork.utils import _convert_input_to_set


def _get_column_dict(name,
                     logical_type,
                     semantic_tags=None,
                     use_standard_tags=True,
                     description=None,
                     metadata=None):
    """Creates a dictionary that contains the typing information for a Schema column
    Args:
        name (str): The name of the column.
        logical_type (str, LogicalType): The column's LogicalType.
        semantic_tags (str, list, set, optional): The semantic tag(s) specified for the column.
        use_standard_tags (boolean, optional): If True, will add standard semantic tags to the column based
                specified logical type. Defaults to True.
        description (str, optional): User description of the column.
        metadata (dict[str -> json serializable], optional): Extra metadata provided by the user.
    """
    _validate_logical_type(logical_type)
    _validate_description(description)

    if metadata is None:
        metadata = {}
    _validate_metadata(metadata)

    semantic_tags = _get_column_tags(semantic_tags, logical_type, use_standard_tags, name)

    return {
        'logical_type': logical_type,
        'semantic_tags': semantic_tags,
        'description': description,
        'metadata': metadata
    }


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
    semantic_tags = _convert_input_to_set(semantic_tags, error_language=f'semantic_tags for {name}')

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


def _add_semantic_tags(new_tags, current_tags, name):
    """Add the specified semantic tags to the current set of tags

    Args:
        new_tags (str/list/set): The new tags to add
        current_tags (set): Current set of semantic tags
        name (str): Name of the column to use in warning
    """
    new_tags = _convert_input_to_set(new_tags)

    duplicate_tags = sorted(list(current_tags.intersection(new_tags)))
    if duplicate_tags:
        warnings.warn(DuplicateTagsWarning().get_warning_message(duplicate_tags, name),
                      DuplicateTagsWarning)
    return current_tags.union(new_tags)


def _remove_semantic_tags(tags_to_remove, current_tags, name, standard_tags, use_standard_tags):
    """Removes specified semantic tags from from the current set of tags

    Args:
        tags_to_remove (str/list/set): The tags to remove
        current_tags (set): Current set of semantic tags
        name (str): Name of the column to use in warning
        standard_tags (set): Set of standard tags for the column logical type
        use_standard_tags (bool): If True, warn if user attempts to remove a standard tag
    """
    tags_to_remove = _convert_input_to_set(tags_to_remove)
    invalid_tags = sorted(list(tags_to_remove.difference(current_tags)))
    if invalid_tags:
        raise LookupError(f"Semantic tag(s) '{', '.join(invalid_tags)}' not present on column '{name}'")
    standard_tags_to_remove = sorted(list(tags_to_remove.intersection(standard_tags)))
    if standard_tags_to_remove and use_standard_tags:
        warnings.warn(StandardTagsChangedWarning().get_warning_message(not use_standard_tags, name),
                      StandardTagsChangedWarning)
    return current_tags.difference(tags_to_remove)


def _reset_semantic_tags(standard_tags, use_standard_tags):
    """Reset the set of semantic tags to the default values. The default values
    will be either an empty set or the standard tags, controlled by the
    use_standard_tags boolean.

    Args:
        standard_tags (set): Set of standard tags for the column logical type
        use_standard_tags (bool): If True, retain standard tags after reset
    """
    if use_standard_tags:
        return set(standard_tags)
    return set()


def _set_semantic_tags(semantic_tags, standard_tags, use_standard_tags):
    """Replace current semantic tags with new values. If use_standard_tags is set
    to True, standard tags will be added as well.

    Args:
        semantic_tags (str/list/set): New semantic tag(s) to set
        standard_tags (set): Set of standard tags for the column logical type
        use_standard_tags (bool): If True, retain standard tags after reset
    """
    semantic_tags = _convert_input_to_set(semantic_tags)

    if use_standard_tags:
        semantic_tags = semantic_tags.union(standard_tags)

    return semantic_tags
