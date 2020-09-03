from data_tables.utils import camel_to_snake


class ClassNameDescriptor(object):
    """Descriptor to convert a class's name from camelcase to snakecase
    """

    def __get__(self, instance, class_):
        return camel_to_snake(class_.__name__)


class LogicalType():
    type_string = ClassNameDescriptor()

    def __repr__(self):
        return u"{}".format('etest')


class Boolean(LogicalType):
    def __repr__(self):
        return "test"


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
    pass


class ZIPCode(LogicalType):
    pass
