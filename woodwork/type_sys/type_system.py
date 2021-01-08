from .inference_functions import (
    boolean_func,
    categorical_func,
    datetime_func,
    double_func,
    integer_func,
    timedelta_func
)

from woodwork.logical_types import (
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
    LogicalType,
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

DEFAULT_TYPE = NaturalLanguage


class TypeSystem(object):
    def __init__(self, inference_functions=None, relationships=None, default_type=NaturalLanguage):
        """Create a new TypeSystem object. LogicalTypes that are present in the keys of
        the inference_functions dictionary will be considered registered LogicalTypes.

        Args:
            inference_functions (dict[LogicalType->func], optional): Dictionary mapping LogicalTypes
                to their corresponding type inference functions. If None, only the default LogicalType
                will be registered without an inference function.
            relationships (list, optional): List of tuples, each with two elements, specifying parent-child
                relationships between logical types. The first element should be the parent LogicalType. The
                second element should be the child LogicalType. If not specified, will default to an empty list
                indicating all types should be considered root types with no children.
            default_type (LogicalType, optional): The default LogicalType to use if no inference matches are
                found. If not specified, will default to the built-in NaturalLanguage LogicalType.
        """
        self.default_type = default_type
        if inference_functions:
            self.inference_functions = inference_functions.copy()
            if self.default_type not in self.inference_functions:
                self.inference_functions[self.default_type] = None
        else:
            self.inference_functions = {self.default_type: None}

        if relationships:
            self.relationships = relationships.copy()
        else:
            self.relationships = []

        # Store initial values for resetting
        self._default_inference_functions = self.inference_functions.copy()
        self._default_relationships = self.relationships.copy()
        self._default_type = self.default_type

    def add_type(self, logical_type, inference_function=None, parent=None):
        """Add a new LogicalType to the TypeSystem, optionally specifying the corresponding inference function and a
        parent type.

        Args:
            logical_type (LogicalType): The new LogicalType to add.
            inference_function (func, optional): The inference function to use for inferring the given LogicalType.
                Defaults to None. If not specified, this LogicalType will never be inferred.
            parent (LogicalType, optional): The parent LogicalType, if applicable. Defaults to None. If not specified,
                this type will be considered a root type with no parent.
        """
        if isinstance(parent, str):
            parent = self.str_to_logical_type(parent)
        self._validate_type_input(logical_type=logical_type,
                                  inference_function=inference_function,
                                  parent=parent)

        registered_ltype_names = [ltype.__name__ for ltype in self.registered_types]
        if logical_type.__name__ in registered_ltype_names:
            raise ValueError(f'Logical Type with name {logical_type.__name__} already present in the Type System. Please rename the LogicalType or remove existing one.')
        self.update_inference_function(logical_type, inference_function)
        if parent:
            self.update_relationship(logical_type, parent)

    def remove_type(self, logical_type):
        """Remove a logical type from the TypeSystem. Any children of the remove type will have their parent
        set to the parent of the removed type.

        Args:
            logical_type (LogicalType): The LogicalType to remove.
        """
        if isinstance(logical_type, str):
            logical_type = self.str_to_logical_type(logical_type)
        self._validate_type_input(logical_type=logical_type)
        # Remove the inference function
        if logical_type == self.default_type:
            raise ValueError("Default LogicalType cannot be removed")
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
        """Update the inference function for the specified LogicalType.

        Args:
            logical_type (LogicalType): The LogicalType for which to update the inference function.
            inference_function (func): The new inference function to use. Can be set to None to skip
                type inference for the specified LogicalType.
        """
        if isinstance(logical_type, str):
            logical_type = self.str_to_logical_type(logical_type)
        self._validate_type_input(logical_type=logical_type, inference_function=inference_function)
        self.inference_functions[logical_type] = inference_function

    def update_relationship(self, logical_type, parent):
        """Add or update a relationship. If the specified LogicalType exists in the relationship graph,
        its parent will be updated. If the specified LogicalType does not exist in relationships, the
        relationship will be added.

        Args:
            logical_type (LogicalType): The LogicalType for which to update the parent value.
            parent (LogicalType): The new parent to set for the specified LogicalType.
        """
        if isinstance(logical_type, str):
            logical_type = self.str_to_logical_type(logical_type)
        if isinstance(parent, str):
            parent = self.str_to_logical_type(parent)
        self._validate_type_input(logical_type=logical_type, parent=parent)
        # If the logical_type already has a parent, remove that from the list
        self.relationships = [rel for rel in self.relationships if rel[1] != logical_type]
        # Add the new/updated relationship
        self.relationships.append((parent, logical_type))

    def reset_defaults(self):
        """Reset type system to the default settings that were specified at initialization.

        Args:
            None
        """
        self.inference_functions = self._default_inference_functions.copy()
        self.relationships = self._default_relationships.copy()
        self.default_type = self._default_type

    @property
    def registered_types(self):
        """Returns a list of all registered types"""
        return list(self.inference_functions.keys())

    @property
    def root_types(self):
        """Returns a list of all registered types that do not have a parent type"""
        return [ltype for ltype in self.registered_types if self._get_parent(ltype) is None]

    def _get_children(self, logical_type):
        """List of all the child types for the given logical type"""
        return [child for parent, child in self.relationships if parent == logical_type]

    def _get_parent(self, logical_type):
        """Get the parent type for the given logical type"""
        for parent, child in self.relationships:
            if child == logical_type:
                return parent
        return None

    def _get_depth(self, logical_type):
        """Get the depth of a type in the relationship graph"""
        depth = 0
        parent = self._get_parent(logical_type)
        while parent:
            depth = depth + 1
            parent = self._get_parent(parent)
        return depth

    def _validate_type_input(self, logical_type=None, inference_function=None, parent=None):
        if logical_type and logical_type not in LogicalType.__subclasses__():
            raise TypeError('logical_type must be a valid LogicalType')

        if inference_function and not callable(inference_function):
            raise TypeError('inference_function must be a function')

        if parent and parent not in self.registered_types:
            raise ValueError('parent must be a valid LogicalType')

    def infer_logical_type(self, series):
        """Infer the logical type for the given series

        Args:
            series (pandas.Series): The series for which to infer the LogicalType.
        """

        # Bring Dask or Koalas data into memory for inference
        if dd and isinstance(series, dd.Series):
            series = series.get_partition(0).compute()
        if ks and isinstance(series, ks.Series):
            series = series.head(100000).to_pandas()

        def get_inference_matches(types_to_check, series, type_matches=[]):
            # Since NaturalLanguage isn't inferred by default, make sure to check
            # any children of NaturalLanguage, otherwise they never get evaluated
            if NaturalLanguage in types_to_check:
                check_next = self._get_children(NaturalLanguage)
            else:
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
            # If no matches, set type to default type (NaturalLanguage)
            return self.default_type
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

    def _get_logical_types(self):
        """Returns a dictionary of logical type name strings and logical type classes"""
        # Get snake case strings
        logical_types = {logical_type.type_string: logical_type for logical_type in self.registered_types}
        # Add class name strings
        class_name_dict = {logical_type.__name__: logical_type for logical_type in self.registered_types}
        logical_types.update(class_name_dict)

        return logical_types

    def str_to_logical_type(self, logical_str, params=None, raise_error=True):
        """Helper function for converting a string value to the corresponding logical type object.
        If a dictionary of params for the logical type is provided, apply them."""
        logical_str_lower = logical_str.lower()
        logical_types_dict = {ltype_name.lower(): ltype for ltype_name, ltype in self._get_logical_types().items()}

        if logical_str_lower in logical_types_dict:
            ltype = logical_types_dict[logical_str_lower]
            if params:
                return ltype(**params)
            else:
                return ltype
        elif raise_error:
            raise ValueError('String %s is not a valid logical type' % logical_str)


type_system = TypeSystem(inference_functions=DEFAULT_INFERENCE_FUNCTIONS,
                         relationships=DEFAULT_RELATIONSHIPS,
                         default_type=DEFAULT_TYPE)
