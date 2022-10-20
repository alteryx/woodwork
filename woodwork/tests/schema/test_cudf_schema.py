import re
from unittest.mock import patch

import pytest

import woodwork as ww
from woodwork.column_schema import (
    ColumnSchema,
    _validate_description,
    _validate_logical_type,
    _validate_metadata,
    _validate_origin,
)
from woodwork.exceptions import DuplicateTagsWarning, StandardTagsChangedWarning
from woodwork.logical_types import (
    Boolean,
    BooleanNullable,
    Categorical,
    Datetime,
    Double,
    Integer,
    IntegerNullable,
    NaturalLanguage,
    Ordinal,
)

from woodwork.utils import (
    _infer_datetime_format,
    _is_valid_latlong_series,
    _is_valid_latlong_value,
    _reformat_to_latlong,
    camel_to_snake,
    import_or_none,
)

cudf = import_or_none("cudf") 


def test_can_init_all_integer_dataframe(): 
    df = cudf.DataFrame()
    df['col1'] = [0, 1, 2, 3]
    df['col2'] = [4, 5, 6, 7]
    df.ww.init(name="cuda") 