import warnings

import woodwork as ww
from woodwork.exceptions import (
    DuplicateTagsWarning,
    StandardTagsChangedWarning
)
from woodwork.logical_types import Boolean, BooleanNullable, Datetime
from woodwork.type_sys.utils import _get_ltype_class
from woodwork.utils import _convert_input_to_set


class ColumnSchema(object):
    def __init__(self,
                 logical_type=None,
                 semantic_tags=None,
                 use_standard_tags=False,
                 description=None,
                 metadata=None,
                 validate=True):
        """Create ColumnSchema

        Args:
            logical_type (LogicalType, optional): The column's LogicalType.
            semantic_tags (str, list, set, optional): The semantic tag(s) specified for the column.
            use_standard_tags (boolean, optional): If True, will add standard semantic tags to the column based
                    on the specified logical type if a logical type is defined for the column. Defaults to False.
            description (str, optional): User description of the column.
            metadata (dict[str -> json serializable], optional): Extra metadata provided by the user.
            validate (bool, optional): Whether to perform parameter validation. Defaults to True.
        """
        if metadata is None:
            metadata = {}
        self.metadata = metadata

        if validate:
            if logical_type is not None:
                _validate_logical_type(logical_type)
            _validate_description(description)
            _validate_metadata(metadata)
        self.description = description
        self.logical_type = logical_type

        self.use_standard_tags = use_standard_tags

        semantic_tags = self._get_column_tags(semantic_tags, validate)
        self.semantic_tags = semantic_tags

    def __eq__(self, other, deep=True):
        if self.use_standard_tags != other.use_standard_tags:
            return False
        if self.logical_type != other.logical_type:
            return False
        if self.semantic_tags != other.semantic_tags:
            return False
        if self.description != other.description:
            return False
        if deep and self.metadata != other.metadata:
            return False

        return True

    def __repr__(self):
        msg = "<ColumnSchema"
        if self.logical_type is not None:
            msg += u" (Logical Type = {})".format(self.logical_type)
        if self.semantic_tags:
            msg += u" (Semantic Tags = {})".format(sorted(list(self.semantic_tags)))
        msg += ">"
        return msg

    def _get_column_tags(self, semantic_tags, validate):
        semantic_tags = _convert_input_to_set(semantic_tags, error_language='semantic_tags',
                                              validate=validate)

        if self.use_standard_tags:
            if self.logical_type is None:
                raise ValueError("Cannot use standard tags when logical_type is None")
            semantic_tags = semantic_tags.union(self.logical_type.standard_tags)

        return semantic_tags

    @property
    def is_numeric(self):
        """Whether the ColumnSchema is numeric in nature"""
        return self.logical_type is not None and 'numeric' in self.logical_type.standard_tags

    @property
    def is_categorical(self):
        """Whether the ColumnSchema is categorical in nature"""
        return self.logical_type is not None and 'category' in self.logical_type.standard_tags

    @property
    def is_datetime(self):
        """Whether the ColumnSchema is a Datetime column"""
        return _get_ltype_class(self.logical_type) == Datetime

    @property
    def is_boolean(self):
        """Whether the ColumnSchema is a Boolean column"""
        ltype_class = _get_ltype_class(self.logical_type)
        return ltype_class == Boolean or ltype_class == BooleanNullable

    def _add_semantic_tags(self, new_tags, name):
        """Add the specified semantic tags to the current set of tags

        Args:
            new_tags (str/list/set): The new tags to add
            name (str): Name of the column to use in warning
        """
        new_tags = _convert_input_to_set(new_tags)

        duplicate_tags = sorted(list(self.semantic_tags.intersection(new_tags)))
        if duplicate_tags:
            warnings.warn(DuplicateTagsWarning().get_warning_message(duplicate_tags, name),
                          DuplicateTagsWarning)
        self.semantic_tags = self.semantic_tags.union(new_tags)

    def _remove_semantic_tags(self, tags_to_remove, name):
        """Removes specified semantic tags from from the current set of tags

        Args:
            tags_to_remove (str/list/set): The tags to remove
            name (str): Name of the column to use in warning
        """
        tags_to_remove = _convert_input_to_set(tags_to_remove)
        invalid_tags = sorted(list(tags_to_remove.difference(self.semantic_tags)))
        if invalid_tags:
            raise LookupError(f"Semantic tag(s) '{', '.join(invalid_tags)}' not present on column '{name}'")

        if self.use_standard_tags and sorted(list(tags_to_remove.intersection(self.logical_type.standard_tags))):
            warnings.warn(StandardTagsChangedWarning().get_warning_message(not self.use_standard_tags, name),
                          StandardTagsChangedWarning)
        self.semantic_tags = self.semantic_tags.difference(tags_to_remove)

    def _reset_semantic_tags(self):
        """Reset the set of semantic tags to the default values. The default values
        will be either an empty set or the standard tags, controlled by the
        use_standard_tags boolean.
        """
        new_tags = set()
        if self.use_standard_tags:
            new_tags = set(self.logical_type.standard_tags)
        self.semantic_tags = new_tags

    def _set_semantic_tags(self, semantic_tags):
        """Replace current semantic tags with new values. If use_standard_tags is set
        to True, standard tags will be added as well.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to set
        """
        semantic_tags = _convert_input_to_set(semantic_tags)

        if self.use_standard_tags:
            semantic_tags = semantic_tags.union(self.logical_type.standard_tags)

        self.semantic_tags = semantic_tags


def _validate_logical_type(logical_type):
    ltype_class = _get_ltype_class(logical_type)

    if ltype_class not in ww.type_system.registered_types:
        raise TypeError(f'logical_type {logical_type} is not a registered LogicalType.')


def _validate_description(column_description):
    if column_description is not None and not isinstance(column_description, str):
        raise TypeError("Column description must be a string")


def _validate_metadata(column_metadata):
    if not isinstance(column_metadata, dict):
        raise TypeError("Column metadata must be a dictionary")
