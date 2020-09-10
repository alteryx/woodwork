from data_tables.utils import camel_to_snake


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

    def __eq__(self, other, deep=False):
        return isinstance(other, self.__class__)


class Boolean(LogicalType):
    pandas_dtype = 'boolean'


class Categorical(LogicalType):
    pandas_dtype = 'category'

    def __init__(self, encoding=None):
        # encoding dict(str -> int)
        # user can specify the encoding to use downstream
        pass


class CountryCode(LogicalType):
    pandas_dtype = 'category'


class Datetime(LogicalType):
    pandas_dtype = 'datetime64[ns]'


class Double(LogicalType):
    pandas_dtype = 'float64'


class Integer(LogicalType):
    pandas_dtype = 'Int64'


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

    def __init__(self, ranking=None):
        # ranking can be used specify the ordering (lowest to highest)
        pass


class PhoneNumber(LogicalType):
    pandas_dtype = 'string'


class SubRegionCode(LogicalType):
    pandas_dtype = 'category'


class Timedelta(LogicalType):
    pandas_dtype = 'timedelta64[ns]'


class URL(LogicalType):
    pandas_dtype = 'string'


class WholeNumber(LogicalType):
    """Represents Logical Types that contain natural numbers, including zero (0)."""
    pandas_dtype = 'Int64'


class ZIPCode(LogicalType):
    pandas_dtype = 'category'


def get_logical_types():
    '''Returns a dictionary of logical type name strings and logical type classes'''
    # Get snake case strings
    logical_types = {logical_type.type_string: logical_type for logical_type in LogicalType.__subclasses__()}
    # Add class name strings
    class_name_dict = {logical_type.__name__: logical_type for logical_type in LogicalType.__subclasses__()}
    logical_types.update(class_name_dict)

    return logical_types
