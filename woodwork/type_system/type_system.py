from .inference_functions import (
    boolean_func,
    categorical_func,
    datetime_func,
    double_func,
    integer_func,
    timedelta_func
)
from .logical_types import (
    URL,
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    EmailAddress,
    Filepath,
    FullName,
    Integer,
    IPAddress,
    LatLong,
    NaturalLanguage,
    Ordinal,
    PhoneNumber,
    SubRegionCode,
    Timedelta,
    ZIPCode
)

from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')

DEFAULT_INFERENCE_FUNCTIONS = {
    Boolean: boolean_func,
    Categorical: categorical_func,
    CountryCode: None,
    Datetime: datetime_func,
    Double: double_func,
    EmailAddress: None,
    Filepath: None,
    FullName: None,
    Integer: integer_func,
    IPAddress: None,
    LatLong: None,
    NaturalLanguage: None,
    Ordinal: None,
    PhoneNumber: None,
    SubRegionCode: None,
    Timedelta: timedelta_func,
    URL: None,
    ZIPCode: None
}

# (ParentType, ChildType)
DEFAULT_RELATIONSHIPS = [
    (Categorical, CountryCode),
    (Categorical, Ordinal),
    (Categorical, SubRegionCode),
    (Categorical, ZIPCode),
    (NaturalLanguage, EmailAddress),
    (NaturalLanguage, Filepath),
    (NaturalLanguage, FullName),
    (NaturalLanguage, IPAddress),
    (NaturalLanguage, PhoneNumber),
    (NaturalLanguage, URL),
]


class TypeSystem(object):
    def __init__(self, inference_functions=None, relationships=None):
        """Add all default logical types and inference functions"""
        self.inference_functions = inference_functions or {}
        self.relationships = relationships or []

    def add_type(self, logical_type, inference_function=None, parent=None):
        """Register a new LogicalType"""
        self.update_inference_function(logical_type, inference_function)
        if parent:
            self.update_relationship(logical_type, parent)

    def remove_type(self, logical_type):
        """Remove a logical type completely"""
        # Remove the inference function
        self.inference_functions.pop(logical_type)

        # If the removed type had children we need to update them
        children = self._get_children(logical_type)
        if children:
            parent = self._get_parent(logical_type)
            for child in children:
                self.update_relationship(child, parent)

        # Rebuild the relationships list to remove any reference to the removed type
        self.relationships = [rel for rel in self.relationships if logical_type not in rel]

    def update_inference_function(self, logical_type, inference_function):
        """Update the inference function for the specified LogicalType"""
        self.inference_functions[logical_type] = inference_function

    def update_relationship(self, logical_type, parent):
        """Add or update a relationship."""
        # If the logical_type already has a parent, remove that from the list
        self.relationships = [rel for rel in self.relationships if rel[1] != logical_type]
        # Add the new/updated relationship
        self.relationships.append((parent, logical_type))

    @property
    def registered_types(self):
        """List of all registered types"""
        return list(self.inference_functions.keys())

    @property
    def root_types(self):
        """List of all types that do not have a parent type"""
        return [ltype for ltype in self.registered_types if self._get_parent(ltype) is None]

    def _get_children(self, logical_type):
        """List of all the child types for the given logical type"""
        return [rel[1] for rel in self.relationships if rel[0] == logical_type]

    def _get_parent(self, logical_type):
        """Get parent for the given logical type"""
        for relationship in self.relationships:
            if relationship[1] == logical_type:
                return relationship[0]
        return None

    def _get_depth(self, logical_type):
        """Get the depth of a type in the relationship graph"""
        depth = 0
        parent = self._get_parent(logical_type)
        while parent:
            depth = depth + 1
            parent = self._get_parent(parent)
        return depth

    def infer_logical_type(self, series):
        """Infer the logical type for the given series"""

        # Bring Dask or Koalas data into memory for inference
        if dd and isinstance(series, dd.Series):
            series = series.get_partition(0).compute()
        if ks and isinstance(series, ks.Series):
            series = series.head(100000).to_pandas()

        def get_inference_matches(types_to_check, series, type_matches=[]):
            check_next = []
            for logical_type in types_to_check:
                inference_func = self.inference_functions.get(logical_type)
                if inference_func and inference_func(series):
                    type_matches.append(logical_type)
                    check_next.extend(self._get_children(logical_type))
            if len(check_next) > 0:
                get_inference_matches(check_next, series, type_matches)
            return type_matches

        type_matches = get_inference_matches(self.root_types, series)

        if len(type_matches) == 0:
            # If no matches, set type to NaturalLanguage
            return NaturalLanguage
        elif len(type_matches) == 1:
            # If we match only one type, return it
            return type_matches[0]
        else:
            # If multiple matches, get the most specific one. If multiple
            # matches have the same level of specificity, the first
            # match found at that level will be returned
            best_match = type_matches[0]
            best_depth = self._get_depth(best_match)
            for logical_type in type_matches[1:]:
                ltype_depth = self._get_depth(logical_type)
                if ltype_depth > best_depth:
                    best_match = logical_type
                    best_depth = ltype_depth
            return best_match


type_sys = TypeSystem(inference_functions=DEFAULT_INFERENCE_FUNCTIONS, relationships=DEFAULT_RELATIONSHIPS)
