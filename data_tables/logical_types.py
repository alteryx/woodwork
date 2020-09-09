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

    def __eq__(self, other, deep=False):
        return isinstance(other, self.__class__)


class Boolean(LogicalType):
    pass


class Categorical(LogicalType):
    def __init__(self, encoding=None):
        # encoding dict(str -> int)
        # user can specify the encoding to use downstream
        pass


class CountryCode(LogicalType):
    pass


class Datetime(LogicalType):
    pass


class Double(LogicalType):
    pass


class Integer(LogicalType):
    pass


class EmailAddress(LogicalType):
    pass


class Filepath(LogicalType):
    pass


class FullName(LogicalType):
    pass


class IPAddress(LogicalType):
    """Represents Logical Types that contain positive, and negative numbers, including zero (0)."""
    pass


class LatLong(LogicalType):
    pass


class NaturalLanguage(LogicalType):
    pass


class Ordinal(LogicalType):
    def __init__(self, ranking=None):
        # ranking can be used specify the ordering (lowest to highest)
        pass


class PhoneNumber(LogicalType):
    pass


class SubRegionCode(LogicalType):
    pass


class Timedelta(LogicalType):
    pass


class URL(LogicalType):
    pass


class WholeNumber(LogicalType):
    """Represents Logical Types that contain natural numbers, including zero (0)."""
    pass


class ZIPCode(LogicalType):
    pass
