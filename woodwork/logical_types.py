from woodwork.utils import camel_to_snake


class ClassNameDescriptor(object):
    """Descriptor to convert a class's name from camelcase to snakecase
    """

    def __get__(self, instance, class_):
        return camel_to_snake(class_.__name__)


class LogicalTypeMetaClass(type):

    def __repr__(cls):
        return cls.__name__


class LogicalType(object, metaclass=LogicalTypeMetaClass):
    type_string = ClassNameDescriptor()
    dtype = 'string'
    standard_tags = {}

    def __eq__(self, other, deep=False):
        return isinstance(other, self.__class__)


class Boolean(LogicalType):
    pandas_dtype = 'boolean'


class Categorical(LogicalType):
    pandas_dtype = 'category'
    standard_tags = {'category'}

    def __init__(self, encoding=None):
        # encoding dict(str -> int)
        # user can specify the encoding to use downstream
        pass


class CountryCode(LogicalType):
    pandas_dtype = 'category'
    standard_tags = {'category'}


class Datetime(LogicalType):
    pandas_dtype = 'datetime64[ns]'


class Double(LogicalType):
    pandas_dtype = 'float64'
    standard_tags = {'numeric'}


class Integer(LogicalType):
    pandas_dtype = 'Int64'
    standard_tags = {'numeric'}


class EmailAddress(LogicalType):
    pandas_dtype = 'string'


class Filepath(LogicalType):
    pandas_dtype = 'string'


class FullName(LogicalType):
    pandas_dtype = 'string'


class IPAddress(LogicalType):
    pandas_dtype = 'string'


class LatLong(LogicalType):
    pandas_dtype = 'string'


class NaturalLanguage(LogicalType):
    pandas_dtype = 'string'


class Ordinal(LogicalType):
    pandas_dtype = 'category'
    standard_tags = {'category'}

    def __init__(self, ranking=None):
        # ranking can be used specify the ordering (lowest to highest)
        pass


class PhoneNumber(LogicalType):
    pandas_dtype = 'string'


class SubRegionCode(LogicalType):
    pandas_dtype = 'category'
    standard_tags = {'category'}


class Timedelta(LogicalType):
    pandas_dtype = 'timedelta64[ns]'


class URL(LogicalType):
    pandas_dtype = 'string'


class WholeNumber(LogicalType):
    """Represents Logical Types that contain natural numbers, including zero (0)."""
    pandas_dtype = 'Int64'
    standard_tags = {'numeric'}


class ZIPCode(LogicalType):
    pandas_dtype = 'category'
    standard_tags = {'category'}


def get_logical_types():
    '''Returns a dictionary of logical type name strings and logical type classes'''
    # Get snake case strings
    logical_types = {logical_type.type_string: logical_type for logical_type in LogicalType.__subclasses__()}
    # Add class name strings
    class_name_dict = {logical_type.__name__: logical_type for logical_type in LogicalType.__subclasses__()}
    logical_types.update(class_name_dict)

    return logical_types


def str_to_logical_type(logical_str, raise_error=True):
    logical_str = logical_str.lower()
    logical_types_dict = {ltype_name.lower(): ltype for ltype_name, ltype in get_logical_types().items()}

    if logical_str in logical_types_dict:
        return logical_types_dict[logical_str]
    elif raise_error:
        raise ValueError('String %s is not a valid logical type' % logical_str)
