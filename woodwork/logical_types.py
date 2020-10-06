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
    into one of a set of possible values. Has 'category' as a standard tag.

    Examples:
        .. code-block:: python

            ["red", "green", "blue"]
            ["produce", "dairy", "bakery"]
            [3, 1, 2]
    """
    pandas_dtype = 'category'
    standard_tags = {'category'}

    def __init__(self, encoding=None):
        # encoding dict(str -> int)
        # user can specify the encoding to use downstream
        pass


class CountryCode(LogicalType):
    """Represents Logical Types that contain categorical information specifically
    used to represent countries. Has 'category' as a standard tag.

    Examples:
        .. code-block:: python

            ["AUS", "USA", "UKR"]
            ["GB", "NZ", "DE"]
    """
    pandas_dtype = 'category'
    standard_tags = {'category'}


class Datetime(LogicalType):
    """Represents Logical Types that contain date and time information.

    Examples:
        .. code-block:: python

            ["2020-09-10",
             "2020-01-10 00:00:00",
             "01/01/2000 08:30"]
    """
    pandas_dtype = 'datetime64[ns]'
    date_format = None

    def __init__(self, date_format=None):
        self.date_format = date_format


class Double(LogicalType):
    """Represents Logical Types that contain positive and negative numbers, some of
    which include a fractional component. Includes zero (0).
    Has 'numeric' as a standard tag.

    Examples:
        .. code-block:: python

            [1.2, 100.4, 3.5]
            [-15.34, 100, 58.3]
    """
    pandas_dtype = 'float64'
    standard_tags = {'numeric'}


class Integer(LogicalType):
    """Represents Logical Types that contain positive and negative numbers
    without a fractional component, including zero (0).
    Has 'numeric' as a standard tag.

    Examples:
        .. code-block:: python

            [100, 35, 0]
            [-54, 73, 11]
    """
    pandas_dtype = 'Int64'
    standard_tags = {'numeric'}


class EmailAddress(LogicalType):
    """Represents Logical Types that contain email address values.

    Examples:
        .. code-block:: python

            ["john.smith@example.com",
             "support@example.com",
             "team@example.com"]
    """
    pandas_dtype = 'string'


class Filepath(LogicalType):
    """Represents Logical Types that specify locations of directories and files
    in a file system.

    Examples:
        .. code-block:: python

            ["/usr/local/bin",
             "/Users/john.smith/dev/index.html",
             "/tmp"]
    """
    pandas_dtype = 'string'


class FullName(LogicalType):
    """Represents Logical Types that may contain first, middle and last names,
    including honorifics and suffixes.

    Examples:
        .. code-block:: python

            ["Mr. John Doe, Jr.",
             "Doe, Mrs. Jane",
             "James Brown"]
    """
    pandas_dtype = 'string'


class IPAddress(LogicalType):
    """Represents Logical Types that contain IP addresses, including both
    IPv4 and IPv6 addresses.

    Examples:
        .. code-block:: python

            ["172.16.254.1",
             "192.0.0.0",
             "2001:0db8:0000:0000:0000:ff00:0042:8329"]
    """
    pandas_dtype = 'string'


class LatLong(LogicalType):
    """Represents Logical Types that contain latitude and longitude values

    Examples:
        .. code-block:: python

            [(33.670914, -117.841501),
             (40.423599, -86.921162)),
             (-45.031705, 168.659506)]
    """
    pandas_dtype = 'string'


class NaturalLanguage(LogicalType):
    """Represents Logical Types that contain text or characters representing
    natural human language

    Examples:
        .. code-block:: python

            ["This is a short sentence.",
             "I like to eat pizza!",
             "When will humans go to mars?"]
    """
    pandas_dtype = 'string'


class Ordinal(LogicalType):
    """Represents Logical Types that contain ordered discrete values.
    Has 'category' as a standard tag.

    Examples:
        .. code-block:: python

            ["first", "second", "third"]
            ["bronze", "silver", "gold"]
    """
    pandas_dtype = 'category'
    standard_tags = {'category'}

    def __init__(self, ranking=None):
        # ranking can be used specify the ordering (lowest to highest)
        pass


class PhoneNumber(LogicalType):
    """Represents Logical Types that contain numeric digits and characters
    representing a phone number

    Examples:
        .. code-block:: python

            ["1-(555)-123-5495",
             "+1-555-123-5495",
             "5551235495"]
    """
    pandas_dtype = 'string'


class SubRegionCode(LogicalType):
    """Represents Logical Types that contain codes representing a portion of
    a larger geographic region. Has 'category' as a standard tag.

    Examples:
        .. code-block:: python

            ["US-CO", "US-MA", "US-CA"]
            ["AU-NSW", "AU-TAS", "AU-QLD"]
    """
    pandas_dtype = 'category'
    standard_tags = {'category'}


class Timedelta(LogicalType):
    """Represents Logical Types that contain values specifying a duration of time

    Examples:
        .. code-block:: python

            [pd.Timedelta('1 days 00:00:00'),
             pd.Timedelta('-1 days +23:40:00'),
             pd.Timedelta('4 days 12:00:00')]
    """
    pandas_dtype = 'timedelta64[ns]'


class URL(LogicalType):
    """Represents Logical Types that contain URLs, which may include protocol, hostname
    and file name

    Examples:
        .. code-block:: python

            ["http://google.com",
             "https://example.com/index.html",
             "example.com"]
    """
    pandas_dtype = 'string'


class WholeNumber(LogicalType):
    """Represents Logical Types that contain natural numbers, including zero (0).
    Has 'numeric' as a standard tag.

    Examples:
        .. code-block:: python

            [3, 30, 56]
            [7, 135, 0]
    """
    pandas_dtype = 'Int64'
    standard_tags = {'numeric'}


class ZIPCode(LogicalType):
    """Represents Logical Types that contain a series of postal codes used by
    the US Postal Service for representing a group of addresses.
    Has 'category' as a standard tag.

    Examples:
        .. code-block:: python

            ["90210"
             "60018-0123",
             "10021"]
    """
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
