import warnings

import pandas as pd

from woodwork.exceptions import (
    DuplicateTagsWarning,
    StandardTagsRemovalWarning
)
from woodwork.logical_types import Ordinal
from woodwork.schema_column import (
    _get_column_dict,
    _validate_description,
    _validate_metadata
)
from woodwork.utils import _convert_input_to_set, _get_column_logical_type


@pd.api.extensions.register_series_accessor('ww')
class WoodworkColumnAccessor:
    def __init__(self, series):
        self._series = series
        self._schema = None

    def init(self, logical_type=None, semantic_tags=None,
             use_standard_tags=True, description=None, metadata=None):
        """Initializes Woodwork typing information for a Series.

        Args:
            logical_type (LogicalType or str, optional): The logical type that should be assigned
                to the series. If no value is provided, the LogicalType for the series will
                be inferred. If the LogicalType provided or inferred does not have a dtype that
                is compatible with the series dtype, an error will be raised.
            semantic_tags (str or list or set, optional): Semantic tags to assign to the series.
                Defaults to an empty set if not specified. There are two options for
                specifying the semantic tags:
                (str) If only one semantic tag is being set, a single string can be passed.
                (list or set) If multiple tags are being set, a list or set of strings can be passed.
            use_standard_tags (bool, optional): If True, will add standard semantic tags to the series
                based on the inferred or specified logical type of the series. Defaults to True.
            description (str, optional): Optional text describing the contents of the series.
            metadata (dict[str -> json serializable], optional): Metadata associated with the series.
        """
        logical_type = _get_column_logical_type(self._series, logical_type, self._series.name)

        self._validate_logical_type(logical_type)
        self.use_standard_tags = use_standard_tags

        self._schema = _get_column_dict(name=self._series.name,
                                        logical_type=logical_type,
                                        semantic_tags=semantic_tags,
                                        use_standard_tags=self.use_standard_tags,
                                        description=description,
                                        metadata=metadata)

    @property
    def description(self):
        """The description of the series"""
        return self._schema['description']

    @description.setter
    def description(self, description):
        _validate_description(description)
        self._schema['description'] = description

    @property
    def logical_type(self):
        """The logical type of the series"""
        return self._schema['logical_type']

    @property
    def metadata(self):
        """The metadata of the series"""
        return self._schema['metadata']

    @metadata.setter
    def metadata(self, metadata):
        _validate_metadata(metadata)
        self._schema['metadata'] = metadata

    @property
    def semantic_tags(self):
        """The semantic tags assigned to the series"""
        return self._schema['semantic_tags']

    def __eq__(self, other):
        if self._schema != other._schema:
            return False
        return self._series.equals(other._series)

    def __repr__(self):
        msg = u"<Series: {} ".format(self._series.name)
        msg += u"(Physical Type = {}) ".format(self._series.dtype)
        msg += u"(Logical Type = {}) ".format(self.logical_type)
        msg += u"(Semantic Tags = {})>".format(self.semantic_tags)
        return msg

    def _validate_logical_type(self, logical_type):
        """Validates that a logical type is consistent with the series dtype. Performs additional type
        specific validation, as required."""
        if logical_type.pandas_dtype != str(self._series.dtype):
            raise ValueError(f"Cannot initialize Woodwork. Series dtype '{self._series.dtype}' is "
                             f"incompatible with {logical_type} dtype. Try converting series "
                             f"dtype to '{logical_type.pandas_dtype}' before initializing.")

        if isinstance(logical_type, Ordinal):
            logical_type._validate_data(self._series)

    def add_semantic_tags(self, semantic_tags):
        """Add the specified semantic tags to the set of tags.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to add
        """
        new_tags = _convert_input_to_set(semantic_tags)

        duplicate_tags = sorted(list(self.semantic_tags.intersection(new_tags)))
        if duplicate_tags:
            warnings.warn(DuplicateTagsWarning().get_warning_message(duplicate_tags, self._series.name),
                          DuplicateTagsWarning)
        updated_tags = self.semantic_tags.union(new_tags)

        self._schema['semantic_tags'] = updated_tags

    def remove_semantic_tags(self, semantic_tags):
        """Removes specified semantic tags from column and returns a new column.

        Args:
            semantic_tags (str/list/set): Semantic tag(s) to remove from the column.
        """
        tags_to_remove = _convert_input_to_set(semantic_tags)
        invalid_tags = sorted(list(tags_to_remove.difference(self.semantic_tags)))
        if invalid_tags:
            raise LookupError(f"Semantic tag(s) '{', '.join(invalid_tags)}' not present on column '{self._series.name}'")
        standard_tags_to_remove = sorted(list(tags_to_remove.intersection(self.logical_type.standard_tags)))
        if standard_tags_to_remove and self.use_standard_tags:
            warnings.warn(StandardTagsRemovalWarning().get_warning_message(standard_tags_to_remove, self._series.name),
                          StandardTagsRemovalWarning)
        new_tags = self.semantic_tags.difference(tags_to_remove)
        self._schema['semantic_tags'] = new_tags

    def reset_semantic_tags(self):
        """Reset the semantic tags to the default values. The default values
        will be either an empty set or a set of the standard tags based on the
        column logical type, controlled by the use_standard_tags property.
        """
        if self.use_standard_tags:
            self._schema['semantic_tags'] = self.logical_type.standard_tags
        else:
            self._schema['semantic_tags'] = set()

    def set_semantic_tags(self, semantic_tags):
        """Replace current semantic tags with new values. If `use_standard_tags` is set
        to True for the series, any standard tags associated with the LogicalType of the
        series will be added as well.

        Args:
            semantic_tags (str/list/set): New semantic tag(s) to set for column
        """
        semantic_tags = _convert_input_to_set(semantic_tags)

        if self.use_standard_tags:
            semantic_tags = semantic_tags.union(self.logical_type.standard_tags)

        self._schema['semantic_tags'] = semantic_tags
