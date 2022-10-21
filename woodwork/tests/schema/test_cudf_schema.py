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
    df["col1"] = [0, 1, 2, 3]
    df["col2"] = [4, 5, 6, 7]
    df.ww.init(name="cuda")


def test_can_init_all_float_dataframe():
    df = cudf.DataFrame()
    df["f1"] = [0.0, 1.0, 2.0, 3.0]
    df["f2"] = [1.0, 2.0, 3.0, 4.0]
    df.ww.init(name="cuda")


def test_can_init_float_and_integer_dataframe():
    df = cudf.DataFrame()
    df["ints"] = [0, 1, 2, 3]
    df["floats"] = [0.0, 1.0, 2.0, 3.0]
    df.ww.init(name="cuda")


def test_can_init_string_dataframe():
    df = cudf.DataFrame()
    df["col1"] = cudf.Series(["a", "b", "c", "d"])
    df.ww.init(name="cuda")


def test_can_infer_email_logical_type():
    df = cudf.DataFrame()
    df["col1"] = cudf.Series(
        [
            "user@gmail.com",
            "user2@gmail.com",
            "user3@gmail.com",
            "user4@gmail.com",
            "user5@gmail.com",
            "user6@gmail.com",
        ]
    )
    df.ww.init(name="cuda")
    assert isinstance(df.ww.logical_types["col1"], ww.logical_types.EmailAddress)


def test_can_check_equality(): 
    df = cudf.DataFrame()
    df["col1"] = [0, 1, 2, 3]
    df["col2"] = [4, 5, 6, 7] 
    df.ww.init(name="cuda") 
    
    df1 = cudf.DataFrame()
    df1["col1"] = [0, 1, 2, 3] 
    df1["col2"] = [4, 5, 6, 7] 
    df1.ww.init(name='cuda') 

    assert df.ww == df1.ww 