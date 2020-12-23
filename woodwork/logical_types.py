import pandas as pd

from woodwork.type_sys.utils import _get_specified_ltype_params
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
    backup_dtype = None
    standard_tags = {}

    def __eq__(self, other, deep=False):
        return isinstance(other, self.__class__) and _get_specified_ltype_params(other) == _get_specified_ltype_params(self)

    def __str__(self):
        return str(self.__class__)


class Boolean(LogicalType):
    """Represents Logical Types that contain binary values indicating true/false.

    Examples:
        .. code-block:: python

            [True, False, True]
            [0, 1, 1]
    """
    pandas_dtype = 'boolean'
    backup_dtype = 'bool'


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
    backup_dtype = 'str'
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
    backup_dtype = 'str'
    standard_tags = {'category'}


class Datetime(LogicalType):
    """Represents Logical Types that contain date and time information.

    Args:
        datetime_format (str): Desired datetime format for data

    Examples:
        .. code-block:: python

            ["2020-09-10",
             "2020-01-10 00:00:00",
             "01/01/2000 08:30"]
    """
    pandas_dtype = 'datetime64[ns]'
    datetime_format = None

    def __init__(self, datetime_format=None):
        self.datetime_format = datetime_format


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
    backup_dtype = 'int64'
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
    backup_dtype = 'str'


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
    backup_dtype = 'str'


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
    backup_dtype = 'str'


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
    backup_dtype = 'str'


class LatLong(LogicalType):
    """Represents Logical Types that contain latitude and longitude values in decimal degrees.

    Note:
        LatLong values will be stored with the object dtype as a
        tuple of floats (or a list of floats for Koalas DataTables)
        and must contain only two values.

        Null latitude or longitude values will be stored as np.nan, and
        a fully null LatLong (np.nan, np.nan) will be stored as just a
        single nan.

    Examples:
        .. code-block:: python

            [(33.670914, -117.841501),
             (40.423599, -86.921162),
             (-45.031705, nan)]
    """
    pandas_dtype = 'object'


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
    backup_dtype = 'str'


class Ordinal(LogicalType):
    """Represents Logical Types that contain ordered discrete values.
    Has 'category' as a standard tag.

    Args:
        order (list or tuple): An list or tuple specifying the order of the ordinal
            values from low to high. The underlying series cannot contain values that
            are not present in the order values.

    Examples:
        .. code-block:: python

            ["first", "second", "third"]
            ["bronze", "silver", "gold"]
    """
    pandas_dtype = 'category'
    backup_dtype = 'str'
    standard_tags = {'category'}

    def __init__(self, order):
        if not isinstance(order, (list, tuple)):
            raise TypeError("Order values must be specified in a list or tuple")
        if len(order) != len(set(order)):
            raise ValueError("Order values cannot contain duplicates")
        self.order = order

    def _validate_data(self, series):
        """Confirm the supplied series does not contain any values that are not
        in the specified order values"""
        if isinstance(series, pd.Series):
            missing_order_vals = set(series.dropna().values).difference(self.order)
            if missing_order_vals:
                error_msg = f'Ordinal column {series.name} contains values that are not present ' \
                    f'in the order values provided: {sorted(list(missing_order_vals))}'
                raise ValueError(error_msg)


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
    backup_dtype = 'str'


class SubRegionCode(LogicalType):
    """Represents Logical Types that contain codes representing a portion of
    a larger geographic region. Has 'category' as a standard tag.

    Examples:
        .. code-block:: python

            ["US-CO", "US-MA", "US-CA"]
            ["AU-NSW", "AU-TAS", "AU-QLD"]
    """
    pandas_dtype = 'category'
    backup_dtype = 'str'
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
    backup_dtype = 'str'


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
    backup_dtype = 'str'
    standard_tags = {'category'}
