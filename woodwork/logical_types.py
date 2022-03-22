import re
import warnings

import pandas as pd
import pandas.api.types as pdtypes

from woodwork.accessor_utils import _is_dask_series, _is_spark_series
from woodwork.config import config
from woodwork.exceptions import (
    TypeConversionError,
    TypeConversionWarning,
    TypeValidationError,
)
from woodwork.type_sys.utils import _get_specified_ltype_params
from woodwork.utils import (
    _infer_datetime_format,
    _is_valid_latlong_series,
    _reformat_to_latlong,
    camel_to_snake,
    import_or_none,
)

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


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
    primary_dtype = "string"
    backup_dtype = None
    standard_tags = set()

    def __eq__(self, other, deep=False):
        return isinstance(other, self.__class__) and _get_specified_ltype_params(
            other
        ) == _get_specified_ltype_params(self)

    def __str__(self):
        return str(self.__class__)

    @classmethod
    def _get_valid_dtype(cls, series_type):
        """Return the dtype that is considered valid for a series with the given logical_type"""
        if ps and series_type == ps.Series and cls.backup_dtype:
            return cls.backup_dtype
        else:
            return cls.primary_dtype

    def transform(self, series):
        """Converts the series dtype to match the logical type's if it is different."""
        new_dtype = self._get_valid_dtype(type(series))
        if new_dtype != str(series.dtype):
            # Update the underlying series
            try:
                series = series.astype(new_dtype)
            except (TypeError, ValueError):
                raise TypeConversionError(series, new_dtype, type(self))
        return series

    def validate(self, series, *args, **kwargs):
        """Validates that a logical type is consistent with the series dtype. Performs additional type
        specific validation, as required. When the series' dtype does not match the logical types' required dtype,
        raises a TypeValidationError."""
        valid_dtype = self._get_valid_dtype(type(series))
        if valid_dtype != str(series.dtype):
            raise TypeValidationError(
                f"Series dtype '{series.dtype}' is incompatible with {self.type_string} dtype."
            )


class Address(LogicalType):
    """Represents Logical Types that contain address values.

    Examples:
        .. code-block:: python

            ['1 Miller Drive, New York, NY 12345', '1 Berkeley Street, Boston, MA 67891']
            ['26387 Russell Hill, Dallas, TX 34521', '54305 Oxford Street, Seattle, WA 95132']
    """

    primary_dtype = "string"


class Age(LogicalType):
    """Represents Logical Types that contain whole numbers indicating a person's age.
    Has 'numeric' as a standard tag.

    Examples:
        .. code-block:: python

            [15, 22, 45]
            [30, 62, 87]
    """

    primary_dtype = "int64"
    standard_tags = {"numeric"}

    def validate(self, series, return_invalid_values=True):
        """Validates age values by checking for non-negative values.

        Args:
            series (Series): Series of age values
            return_invalid_values (bool): Whether or not to return invalid age values

        Returns:
            Series: If return_invalid_values is True, returns invalid age values.
        """
        return _validate_not_negative(series, return_invalid_values)


class AgeFractional(LogicalType):
    """Represents Logical Types that contain non-negative floating point numbers indicating a person's age.
    Has 'numeric' as a standard tag. May also contain null values.

    Examples:
        .. code-block:: python

            [0.34, 24.34, 45.0]
            [30.5, 62.82, np.nan]
    """

    primary_dtype = "float64"
    standard_tags = {"numeric"}

    def validate(self, series, return_invalid_values=True):
        """Validates age values by checking for non-negative values.

        Args:
            series (Series): Series of age values
            return_invalid_values (bool): Whether or not to return invalid age values

        Returns:
            Series: If return_invalid_values is True, returns invalid age values.
        """
        return _validate_not_negative(series, return_invalid_values)


class AgeNullable(LogicalType):
    """Represents Logical Types that contain whole numbers indicating a person's age.
    Has 'numeric' as a standard tag. May also contain null values.

    Examples:
        .. code-block:: python

            [np.nan, 22, 45]
            [30, 62, np.nan]
    """

    primary_dtype = "Int64"
    standard_tags = {"numeric"}

    def validate(self, series, return_invalid_values=True):
        """Validates age values by checking for non-negative values.

        Args:
            series (Series): Series of age values
            return_invalid_values (bool): Whether or not to return invalid age values

        Returns:
            Series: If return_invalid_values is True, returns invalid age values.
        """
        return _validate_not_negative(series, return_invalid_values)


class Boolean(LogicalType):
    """Represents Logical Types that contain binary values indicating true/false.

    Examples:
        .. code-block:: python

            [True, False, True]
            [0, 1, 1]
    """

    primary_dtype = "bool"


class BooleanNullable(LogicalType):
    """Represents Logical Types that contain binary values indicating true/false.
    May also contain null values.

    Examples:
        .. code-block:: python

            [True, False, None]
            [0, 1, 1]
    """

    primary_dtype = "boolean"


class Categorical(LogicalType):
    """Represents Logical Types that contain unordered discrete values that fall
    into one of a set of possible values. Has 'category' as a standard tag.

    Examples:
        .. code-block:: python

            ["red", "green", "blue"]
            ["produce", "dairy", "bakery"]
            [3, 1, 2]
    """

    primary_dtype = "category"
    backup_dtype = "string"
    standard_tags = {"category"}

    def __init__(self, encoding=None):
        # encoding dict(str -> int)
        # user can specify the encoding to use downstream
        pass


class CountryCode(LogicalType):
    """Represents Logical Types that use the ISO-3166 standard country code to represent countries.
    ISO 3166-1 (countries) are supported. These codes should be in the Alpha-2 format.

    Examples:
        .. code-block:: python

            ["AU", "US", "UA"]
            ["GB", "NZ", "DE"]
    """

    primary_dtype = "category"
    backup_dtype = "string"
    standard_tags = {"category"}


class CurrencyCode(LogicalType):
    """Represents Logical Types that use the ISO-4217 internation standard currency code to represent currencies.

    Examples:
        .. code-block:: python

            ["GBP", "JPY", "USD"]
            ["SAR", "EUR", "CZK"]
    """

    primary_dtype = "category"
    backup_dtype = "string"
    standard_tags = {"category"}


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

    primary_dtype = "datetime64[ns]"
    datetime_format = None

    def __init__(self, datetime_format=None):
        self.datetime_format = datetime_format

    def transform(self, series):
        """Converts the series data to a formatted datetime. Datetime format will be inferred if datetime_format is None."""
        new_dtype = self._get_valid_dtype(type(series))
        series_dtype = str(series.dtype)

        if new_dtype != series_dtype:
            self.datetime_format = self.datetime_format or _infer_datetime_format(
                series
            )
            if _is_dask_series(series):
                name = series.name
                series = dd.to_datetime(
                    series, format=self.datetime_format, errors="coerce"
                )
                series.name = name
            elif _is_spark_series(series):
                series = ps.Series(
                    ps.to_datetime(
                        series.to_numpy(), format=self.datetime_format, errors="coerce"
                    ),
                    name=series.name,
                )
            else:
                try:
                    series = pd.to_datetime(series, format=self.datetime_format)
                except (TypeError, ValueError):
                    warnings.warn(
                        f"Some rows in series '{series.name}' are incompatible with datetime format "
                        f"'{self.datetime_format}' and have been replaced with null values. You may be "
                        "able to fix this by using an instantiated Datetime logical type with a different format "
                        "string specified for this column during Woodwork initialization.",
                        TypeConversionWarning,
                    )
                    series = pd.to_datetime(
                        series, format=self.datetime_format, errors="coerce"
                    )
        return super().transform(series)


class Double(LogicalType):
    """Represents Logical Types that contain positive and negative numbers, some of
    which include a fractional component. Includes zero (0).
    Has 'numeric' as a standard tag.

    Examples:
        .. code-block:: python

            [1.2, 100.4, 3.5]
            [-15.34, 100, 58.3]
    """

    primary_dtype = "float64"
    standard_tags = {"numeric"}


class Integer(LogicalType):
    """Represents Logical Types that contain positive and negative numbers
    without a fractional component, including zero (0).
    Has 'numeric' as a standard tag.

    Examples:
        .. code-block:: python

            [100, 35, 0]
            [-54, 73, 11]
    """

    primary_dtype = "int64"
    standard_tags = {"numeric"}


class IntegerNullable(LogicalType):
    """Represents Logical Types that contain positive and negative numbers
    without a fractional component, including zero (0). May contain null
    values. Has 'numeric' as a standard tag.

    Examples:
        .. code-block:: python

            [100, 35, np.nan]
            [-54, 73, 11]
    """

    primary_dtype = "Int64"
    standard_tags = {"numeric"}


class EmailAddress(LogicalType):
    """Represents Logical Types that contain email address values.

    Examples:
        .. code-block:: python

            ["john.smith@example.com",
             "support@example.com",
             "team@example.com"]
    """

    primary_dtype = "string"

    def validate(self, series, return_invalid_values=False):
        """Validates email address values based on the regex in the config.

        Args:
            series (Series): Series of email address values
            return_invalid_values (bool): Whether or not to return invalid email address values

        Returns:
            Series: If return_invalid_values is True, returns invalid email address.
        """
        return _regex_validate("email_inference_regex", series, return_invalid_values)


class Filepath(LogicalType):
    """Represents Logical Types that specify locations of directories and files
    in a file system.

    Examples:
        .. code-block:: python

            ["/usr/local/bin",
             "/Users/john.smith/dev/index.html",
             "/tmp"]
    """

    primary_dtype = "string"


class PersonFullName(LogicalType):
    """Represents Logical Types that may contain first, middle and last names,
    including honorifics and suffixes.

    Examples:
        .. code-block:: python

            ["Mr. John Doe, Jr.",
             "Doe, Mrs. Jane",
             "James Brown"]
    """

    primary_dtype = "string"


class IPAddress(LogicalType):
    """Represents Logical Types that contain IP addresses, including both
    IPv4 and IPv6 addresses.

    Examples:
        .. code-block:: python

            ["172.16.254.1",
             "192.0.0.0",
             "2001:0db8:0000:0000:0000:ff00:0042:8329"]
    """

    primary_dtype = "string"


class LatLong(LogicalType):
    """Represents Logical Types that contain latitude and longitude values in decimal degrees.

    Note:
        LatLong values will be stored with the object dtype as a
        tuple of floats (or a list of floats for Spark DataFrames)
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

    primary_dtype = "object"

    def transform(self, series):
        """Formats a series to be a tuple (or list for Spark) of two floats."""
        if _is_dask_series(series):
            name = series.name
            meta = (series, tuple([float, float]))
            series = series.apply(_reformat_to_latlong, meta=meta)
            series.name = name
        elif _is_spark_series(series):
            formatted_series = series.to_pandas().apply(
                _reformat_to_latlong, is_spark=True
            )
            series = ps.from_pandas(formatted_series)
        else:
            series = series.apply(_reformat_to_latlong)

        return super().transform(series)

    def validate(self, series, return_invalid_values=False):
        # TODO: we'll want to actually handle return_invalid_values in the ordinal and latlong logical types.
        super().validate(series)
        if not _is_valid_latlong_series(series):
            raise TypeValidationError(
                "Cannot initialize Woodwork. Series does not contain properly formatted "
                "LatLong data. Try reformatting before initializing or use the "
                "woodwork.init_series function to initialize."
            )


class NaturalLanguage(LogicalType):
    """Represents Logical Types that contain text or characters representing
    natural human language

    Examples:
        .. code-block:: python

            ["This is a short sentence.",
             "I like to eat pizza!",
             "When will humans go to mars?"]
    """

    primary_dtype = "string"


class Unknown(LogicalType):
    """Represents Logical Types that cannot be inferred as a specific Logical Type. It is assumed to contain string data.

    Examples:
        .. code-block:: python

            ["ax23n9ck23l",
             "1,28&*_%*&&xejc",
             "xnmvz@@Dcmeods-0"]

    """

    primary_dtype = "string"


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

    primary_dtype = "category"
    backup_dtype = "string"
    standard_tags = {"category"}

    def __init__(self, order=None):
        self.order = order

    def _validate_order_values(self, series):
        """Make sure order values are properly defined and confirm the supplied series
        does not contain any values that are not in the specified order values"""
        if self.order is None:
            raise TypeError("Must use an Ordinal instance with order values defined")
        elif not isinstance(self.order, (list, tuple)):
            raise TypeError("Order values must be specified in a list or tuple")
        if len(self.order) != len(set(self.order)):
            raise ValueError("Order values cannot contain duplicates")

        if isinstance(series, pd.Series):
            missing_order_vals = set(series.dropna().values).difference(self.order)
            if missing_order_vals:
                error_msg = (
                    f"Ordinal column {series.name} contains values that are not present "
                    f"in the order values provided: {sorted(list(missing_order_vals))}"
                )
                raise ValueError(error_msg)

    def transform(self, series):
        """Validates the series and converts the dtype to match the logical type's if it is different."""
        self._validate_order_values(series)

        typed_ser = super().transform(series)
        if pdtypes.is_categorical_dtype(typed_ser.dtype):
            typed_ser = typed_ser.cat.set_categories(self.order, ordered=True)

        return typed_ser

    def validate(self, series, return_invalid_values=False):
        # TODO: we'll want to actually handle return_invalid_values in the ordinal and latlong logical types.
        super().validate(series)
        self._validate_order_values(series)

    def __str__(self):
        return "{}: {}".format(self.__class__, self.order)


class PhoneNumber(LogicalType):
    """Represents Logical Types that contain numeric digits and characters
    representing a phone number

    Examples:
        .. code-block:: python

            ["1-(555)-123-5495",
             "+1-555-123-5495",
             "5551235495"]
    """

    primary_dtype = "string"


class SubRegionCode(LogicalType):
    """Represents Logical Types that use the ISO-3166 standard sub-region code to
    represent a portion of a larger geographic region. ISO 3166-2 (sub-regions)
    codes are supported. These codes should be in the Alpha-2 format.

    Examples:
        .. code-block:: python

            ["US-CO", "US-MA", "US-CA"]
            ["AU-NSW", "AU-TAS", "AU-QLD"]
    """

    primary_dtype = "category"
    backup_dtype = "string"
    standard_tags = {"category"}


class Timedelta(LogicalType):
    """Represents Logical Types that contain values specifying a duration of time

    Examples:
        .. code-block:: python

            [pd.Timedelta('1 days 00:00:00'),
             pd.Timedelta('-1 days +23:40:00'),
             pd.Timedelta('4 days 12:00:00')]
    """

    primary_dtype = "timedelta64[ns]"


class URL(LogicalType):
    """Represents Logical Types that contain URLs, which may include protocol, hostname
    and file name

    Examples:
        .. code-block:: python

            ["http://google.com",
             "https://example.com/index.html",
             "example.com"]
    """

    primary_dtype = "string"

    def validate(self, series, return_invalid_values=False):
        """Validates URL values based on the regex in the config.

        Args:
            series (Series): Series of URL values
            return_invalid_values (bool): Whether or not to return invalid URLs

        Returns:
            Series: If return_invalid_values is True, returns invalid URLs.
        """
        return _regex_validate("url_inference_regex", series, return_invalid_values)


class PostalCode(LogicalType):
    """Represents Logical Types that contain a series of postal codes for
    representing a group of addresses. Has 'category' as a standard tag.

    Examples:
        .. code-block:: python

            ["90210"
             "60018-0123",
             "SW1A"]
    """

    primary_dtype = "category"
    backup_dtype = "string"
    standard_tags = {"category"}


_NULLABLE_PHYSICAL_TYPES = {
    "boolean",
    "category",
    "datetime64[ns]",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Float32",
    "Float64",
    "float16",
    "float32",
    "float64",
    "float128",
    "object",
    "string",
    "timedelta64[ns]",
}


def _regex_validate(regex_key, series, return_invalid_values):
    """Validates data values based on the logical type regex in the config."""
    regex = config.get_option(regex_key)

    if _is_spark_series(series):

        def match(x):
            if isinstance(x, str):
                return bool(re.match(regex, x))

        invalid = series.apply(match).astype("boolean") == False  # noqa: E712

    else:
        invalid = ~series.str.match(regex).astype("boolean")

    if return_invalid_values:
        return series[invalid]

    else:
        any_invalid = invalid.any()
        if dd and isinstance(any_invalid, dd.core.Scalar):
            any_invalid = any_invalid.compute()

        if any_invalid:
            type_string = {
                "url_inference_regex": "url",
                "email_inference_regex": "email address",
            }[regex_key]

            info = f"Series {series.name} contains invalid {type_string} values. "
            info += f"The {regex_key} can be changed in the config if needed."
            raise TypeValidationError(info)


def _validate_not_negative(series, return_invalid_values):
    """Validates data values are non-negative."""
    invalid = series.lt(0)
    if return_invalid_values:
        return series[invalid]

    else:
        any_invalid = invalid.any()
        if dd and isinstance(any_invalid, dd.core.Scalar):
            any_invalid = any_invalid.compute()

        if any_invalid:
            info = f"Series {series.name} contains negative values."
            raise TypeValidationError(info)
