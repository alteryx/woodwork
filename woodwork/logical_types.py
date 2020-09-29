from woodwork.utils import camel_to_snake


class ClassNameDescriptor(object):
    """Descriptor to convert a class's name from camelcase to snakecase"""

    def __get__(self, instance, class_):
        return camel_to_snake(class_.__name__)


class LogicalTypeMetaClass(type):

    def __repr__(cls):
        return cls.__name__


class LogicalType(object, metaclass=LogicalTypeMetaClass):
    """Base class for all other Logical Types"""
    type_string = ClassNameDescriptor()
    pandas_dtype = 'string'
    standard_tags = {}

    def __eq__(self, other, deep=False):
        return isinstance(other, self.__class__)


class Boolean(LogicalType):
    """Represents Logical Types that contain binary values indicating true/false.

    Examples:
        .. code-block:: python

            [True, False, True]
            [0, 1, 1]
    """
    pandas_dtype = 'boolean'


class Categorical(LogicalType):
    """Represents Logical Types that contain unordered discrete values that fall
    into one of a set of possible values
    
    Examples:
        .. code-block:: python

            ["red", "green", "blue"]
            ["produce", "dairy", "bakery"]

    """
    pandas_dtype = 'category'
    standard_tags = {'category'}

    def __init__(self, encoding=None):
        # encoding dict(str -> int)
        # user can specify the encoding to use downstream
        pass


class CountryCode(LogicalType):
    """Represents Logical Types that contain categorical information specifically
    used to represent countries"""
    pandas_dtype = 'category'
    standard_tags = {'category'}


class Datetime(LogicalType):
    """Represents Logical Types that contain date and time information"""
    pandas_dtype = 'datetime64[ns]'


class Double(LogicalType):
    """Represents Logical Types that contain positive and negative numbers that
    may include a fractional component, including zero (0)"""
    pandas_dtype = 'float64'
    standard_tags = {'numeric'}


class Integer(LogicalType):
    """Represents Logical Types that contain positive and negative numbers
    without a fractional component, including zero (0)"""
    pandas_dtype = 'Int64'
    standard_tags = {'numeric'}


class EmailAddress(LogicalType):
    """Represents Logical Types that contain email address values"""
    pandas_dtype = 'string'


class Filepath(LogicalType):
    """Represents Logical Types that specify locations of files in a file system"""
    pandas_dtype = 'string'


class FullName(LogicalType):
    """Represents Logical Types that may contain first, middle and last names,
    including honorifics and suffixes"""
    pandas_dtype = 'string'


class IPAddress(LogicalType):
    """Represents Logical Types that contain IP addresses, including both
    IPv4 and IPv6 addresses"""
    pandas_dtype = 'string'


class LatLong(LogicalType):
    """Represents Logical Types that contain lattitude and longitude values"""
    pandas_dtype = 'string'


class NaturalLanguage(LogicalType):
    """Represents Logical Types that contain text or characters representing
    natural human language"""
    pandas_dtype = 'string'


class Ordinal(LogicalType):
    """Represents Logical Types that contain ordered discrete values"""
    pandas_dtype = 'category'
    standard_tags = {'category'}

    def __init__(self, ranking=None):
        # ranking can be used specify the ordering (lowest to highest)
        pass


class PhoneNumber(LogicalType):
    """Represents Logical Types that contain numeric digits and characters
    representing a phone number"""
    pandas_dtype = 'string'


class SubRegionCode(LogicalType):
    """Represents Logical Types that contain codes representing a portion of
    a larger geographic region"""
    pandas_dtype = 'category'
    standard_tags = {'category'}


class Timedelta(LogicalType):
    """Represents Logical Types that contain values specifying a duration of time"""
    pandas_dtype = 'timedelta64[ns]'


class URL(LogicalType):
    """Represents Logical Types that contain URLs, which may include protocal, hostname
    and file name"""
    pandas_dtype = 'string'


class WholeNumber(LogicalType):
    """Represents Logical Types that contain natural numbers, including zero (0)."""
    pandas_dtype = 'Int64'
    standard_tags = {'numeric'}


class ZIPCode(LogicalType):
    """Represents Logical Types that contain a series of digits used for sorting mail"""
    pandas_dtype = 'category'
    standard_tags = {'category'}


def get_logical_types():
    """Returns a dictionary of logical type name strings and logical type classes"""
    # Get snake case strings
    logical_types = {logical_type.type_string: logical_type for logical_type in LogicalType.__subclasses__()}
    # Add class name strings
    class_name_dict = {logical_type.__name__: logical_type for logical_type in LogicalType.__subclasses__()}
    logical_types.update(class_name_dict)

    return logical_types


def str_to_logical_type(logical_str, raise_error=True):
    """Helper function for converting a string value to the corresponding logical type object"""
    logical_str = logical_str.lower()
    logical_types_dict = {ltype_name.lower(): ltype for ltype_name, ltype in get_logical_types().items()}

    if logical_str in logical_types_dict:
        return logical_types_dict[logical_str]
    elif raise_error:
        raise ValueError('String %s is not a valid logical type' % logical_str)
