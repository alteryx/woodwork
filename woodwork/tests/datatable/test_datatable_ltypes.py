import re

import pandas as pd
import pytest

import woodwork as ww
from woodwork import DataTable
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    FullName,
    Integer,
    LogicalType,
    NaturalLanguage,
    PhoneNumber,
    ZIPCode
)
from woodwork.tests.testing_utils import to_pandas


def test_datatable_ltypes(sample_df):
    dt = DataTable(sample_df)
    returned_types = dt.ltypes
    assert isinstance(returned_types, pd.Series)
    assert returned_types.name == 'Logical Type'
    assert len(returned_types.index) == len(sample_df.columns)
    assert all([issubclass(logical_type, LogicalType) for logical_type in returned_types.values])
    correct_logical_types = {
        'id': Integer,
        'full_name': NaturalLanguage,
        'email': NaturalLanguage,
        'phone_number': NaturalLanguage,
        'age': Integer,
        'signup_date': Datetime,
        'is_registered': Boolean
    }
    correct_logical_types = pd.Series(list(correct_logical_types.values()),
                                      index=list(correct_logical_types.keys()))
    assert correct_logical_types.equals(returned_types)


def test_datatable_logical_types(sample_df):
    dt = DataTable(sample_df)
    assert isinstance(dt.logical_types, dict)
    assert set(dt.logical_types.keys()) == set(sample_df.columns)
    for k, v in dt.logical_types.items():
        assert isinstance(k, str)
        assert k in sample_df.columns
        assert v in ww.type_system.registered_types
        assert v == dt.columns[k].logical_type


def test_set_logical_types(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'phone_number': ['tag3', 'tag2'],
        'signup_date': {'secondary_time_index'},
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags, use_standard_tags=True)

    new_dt = dt.set_types(logical_types={
        'full_name': Categorical,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
    })

    # Verify original types were not changed
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['email'].logical_type == NaturalLanguage
    assert dt.columns['phone_number'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == Integer
    assert dt.columns['signup_date'].logical_type == Datetime
    original_name_column = dt.columns['full_name']

    assert new_dt is not dt
    assert new_dt.columns['full_name'].logical_type == Categorical
    assert new_dt.columns['email'].logical_type == EmailAddress
    assert new_dt.columns['phone_number'].logical_type == PhoneNumber
    assert new_dt.columns['age'].logical_type == Double

    # Verify new column object was created
    new_name_column = new_dt.columns['full_name']
    assert new_name_column is not original_name_column

    # Verify semantic tags were reset to standard tags
    assert new_dt.columns['full_name'].semantic_tags == {'category'}
    assert new_dt.columns['email'].semantic_tags == set()
    assert new_dt.columns['phone_number'].semantic_tags == set()
    assert new_dt.columns['age'].semantic_tags == {'numeric'}

    # Verify signup date column was unchanged
    assert new_dt.columns['signup_date'].logical_type == Datetime
    assert new_dt.columns['signup_date'].semantic_tags == {'secondary_time_index'}


def test_set_logical_types_invalid_data(sample_df):
    dt = DataTable(sample_df)
    error_message = re.escape("logical_types contains columns that are not present in dataframe: ['birthday']")
    with pytest.raises(LookupError, match=error_message):
        dt.set_types(logical_types={'birthday': Double})

    error_message = "Invalid logical type specified for 'age'"
    with pytest.raises(TypeError, match=error_message):
        dt.set_types(logical_types={'age': int})


def test_select_ltypes_no_match_and_all(sample_df):
    dt = DataTable(sample_df)
    dt = dt.set_types(logical_types={
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })
    assert len(dt.select(ZIPCode).columns) == 0
    assert len(dt.select(['ZIPCode', PhoneNumber]).columns) == 1
    all_types = ww.type_system.registered_types
    dt_all_types = dt.select(all_types)
    assert len(dt_all_types.columns) == len(dt.columns)
    assert len(dt_all_types.to_dataframe().columns) == len(dt.to_dataframe().columns)


def test_select_ltypes_strings(sample_df):
    dt = DataTable(sample_df)
    dt = dt.set_types(logical_types={
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })

    dt_multiple_ltypes = dt.select(['FullName', 'email_address', 'double', 'Boolean', 'datetime'])
    assert len(dt_multiple_ltypes.columns) == 5
    assert 'phone_number' not in dt_multiple_ltypes.columns
    assert 'id' not in dt_multiple_ltypes.columns

    dt_single_ltype = dt.select('full_name')
    assert len(dt_single_ltype.columns) == 1


def test_select_ltypes_objects(sample_df):
    dt = DataTable(sample_df)
    dt = dt.set_types(logical_types={
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })

    dt_multiple_ltypes = dt.select([FullName, EmailAddress, Double, Boolean, Datetime])
    assert len(dt_multiple_ltypes.columns) == 5
    assert 'phone_number' not in dt_multiple_ltypes.columns
    assert 'id' not in dt_multiple_ltypes.columns

    dt_single_ltype = dt.select(FullName)
    assert len(dt_single_ltype.columns) == 1


def test_select_ltypes_mixed(sample_df):
    dt = DataTable(sample_df)
    dt = dt.set_types(logical_types={
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })

    dt_mixed_ltypes = dt.select(['FullName', 'email_address', Double])
    assert len(dt_mixed_ltypes.columns) == 3
    assert 'phone_number' not in dt_mixed_ltypes.columns


def test_select_ltypes_table(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id')
    dt = dt.set_types(logical_types={
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })
    dt.set_types(semantic_tags={
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
    })

    dt_no_indices = dt.select('phone_number')
    assert dt_no_indices.index is None
    assert dt_no_indices.time_index is None

    dt_with_indices = dt.select(['Datetime', 'Integer'])
    assert dt_with_indices.index == 'id'
    assert dt_with_indices.time_index == 'signup_date'

    dt_values = dt.select(['FullName'])
    assert dt_values.name == dt.name
    original_col = dt_values.columns['full_name']
    col = dt.columns['full_name']
    assert col.logical_type == original_col.logical_type
    assert to_pandas(col.to_series()).equals(to_pandas(original_col.to_series()))
    assert col.dtype == original_col.dtype
    assert col.semantic_tags == original_col.semantic_tags
