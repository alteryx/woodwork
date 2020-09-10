import re

import numpy as np
import pandas as pd
import pytest

from data_tables import DataTable
from data_tables.data_table import (
    _check_index,
    _check_logical_types,
    _check_semantic_types,
    _check_time_index,
    _check_unique_column_names,
    _validate_params
)
from data_tables.logical_types import (
    URL,
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    EmailAddress,
    FilePath,
    FullName,
    Integer,
    IPAddress,
    LatLong,
    LogicalType,
    NaturalLanguage,
    Ordinal,
    PhoneNumber,
    SubRegionCode,
    WholeNumber,
    ZIPCode
)


def test_datatable_init(sample_df):
    dt = DataTable(sample_df)
    df = dt.df

    assert dt.name is None
    assert dt.index is None
    assert dt.time_index is None
    assert isinstance(df, pd.DataFrame)
    assert set(dt.columns.keys()) == set(sample_df.columns)
    assert df is sample_df
    pd.testing.assert_frame_equal(df, sample_df)


def test_datatable_copy_param(sample_df):
    dt_with_copy = DataTable(sample_df, copy_dataframe=True)
    assert sample_df is not dt_with_copy.df

    dt_no_copy = DataTable(sample_df)
    assert sample_df is dt_no_copy.df


def test_datatable_init_with_name_and_index_vals(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   index='id',
                   time_index='signup_date')

    assert dt.name == 'datatable'
    assert dt.index == 'id'
    assert dt.time_index == 'signup_date'


def test_datatable_init_with_logical_types(sample_df):
    logical_types = {
        'full_name': NaturalLanguage,
        'age': Double
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types=logical_types)
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == Double


def test_datatable_init_with_string_logical_types(sample_df):
    logical_types = {
        'full_name': 'natural_language',
        'age': 'whole_number'
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types=logical_types)
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == WholeNumber

    logical_types = {
        'full_name': 'NaturalLanguage',
        'age': 'WholeNumber'
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types=logical_types)
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == WholeNumber


def test_datatable_init_with_semantic_types(sample_df):
    semantic_types = {
        'id': 'index',
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   semantic_types=semantic_types)

    id_semantic_types = dt.columns['id'].semantic_types
    assert isinstance(id_semantic_types, dict)
    assert len(id_semantic_types) == 1
    assert 'index' in id_semantic_types.keys()
    assert id_semantic_types['index'] == {}


def test_validate_params_errors(sample_df):
    error_message = 'Dataframe must be a pandas.DataFrame'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=pd.Series(),
                         name=None,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         semantic_types=None)

    error_message = 'DataTable name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=sample_df,
                         name=1,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         semantic_types=None)


def test_check_index_errors(sample_df):
    error_message = 'Index column name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _check_index(dataframe=sample_df, index=1)

    error_message = 'Specified index column `foo` not found in dataframe'
    with pytest.raises(LookupError, match=error_message):
        _check_index(dataframe=sample_df, index='foo')

    error_message = 'Index column must be unique'
    with pytest.raises(LookupError, match=error_message):
        _check_index(sample_df, index='age')


def test_check_time_index_errors(sample_df):
    error_message = 'Time index column name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _check_time_index(dataframe=sample_df,
                          time_index=1)

    error_message = 'Specified time index column `foo` not found in dataframe'
    with pytest.raises(LookupError, match=error_message):
        _check_time_index(dataframe=sample_df, time_index='foo')


def test_check_unique_column_names(sample_df):
    duplicate_cols_df = sample_df.copy()
    duplicate_cols_df.insert(0, 'age', [18, 21, 65], allow_duplicates=True)
    with pytest.raises(IndexError, match='Dataframe cannot contain duplicate columns names'):
        _check_unique_column_names(duplicate_cols_df)


def test_check_logical_types_errors(sample_df):
    error_message = 'logical_types must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_logical_types(sample_df, logical_types='type')

    bad_logical_types_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("logical_types contains columns that are not present in dataframe: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_logical_types(sample_df, bad_logical_types_keys)


def test_datatable_types(sample_df):
    dt = DataTable(sample_df)
    returned_types = dt.types
    assert isinstance(returned_types, pd.DataFrame)
    assert 'Physical Type' in returned_types.columns
    assert 'Logical Type' in returned_types.columns
    assert 'Semantic Tag(s)' in returned_types.columns
    assert returned_types.shape[1] == 3
    assert len(returned_types.index) == len(sample_df.columns)
    assert all([issubclass(dc.logical_type, LogicalType) for dc in dt.columns.values()])
    correct_logical_types = {
        'id': WholeNumber,
        'full_name': NaturalLanguage,
        'email': NaturalLanguage,
        'phone_number': NaturalLanguage,
        'age': WholeNumber,
        'signup_date': Datetime,
        'is_registered': Boolean
    }
    correct_logical_types = pd.Series(list(correct_logical_types.values()),
                                      index=list(correct_logical_types.keys()))
    assert correct_logical_types.equals(returned_types['Logical Type'])
    for tag in returned_types['Semantic Tag(s)']:
        assert isinstance(tag, dict)
        # TODO: Add a tag to DataTable, and check the tag shows up
        # Waiting on init with semantic tags / set_semantic_tags


def test_datatable_physical_types(sample_df):
    dt = DataTable(sample_df)
    assert isinstance(dt.physical_types, dict)
    assert set(dt.physical_types.keys()) == set(sample_df.columns)
    for k, v in dt.physical_types.items():
        assert isinstance(k, str)
        assert v == sample_df[k].dtype


def test_datatable_logical_types(sample_df):
    dt = DataTable(sample_df)
    assert isinstance(dt.logical_types, dict)
    assert set(dt.logical_types.keys()) == set(sample_df.columns)
    for k, v in dt.logical_types.items():
        assert isinstance(k, str)
        assert k in sample_df.columns
        assert v in LogicalType.__subclasses__()
        assert v == dt.columns[k].logical_type


def test_datatable_semantic_types(sample_df):
    semantic_types = {
        'full_name': 'tag1',
        'email': {'tag2': {'option1': 'value1'}},
        'phone_number': {'tag3': None},
        'signup_date': {'secondary_time_index': {'columns': ['expired']}},
        'age': ['numeric', 'age']
    }
    dt = DataTable(sample_df, semantic_types=semantic_types)
    assert isinstance(dt.semantic_types, dict)
    assert set(dt.semantic_types.keys()) == set(sample_df.columns)
    for k, v in dt.semantic_types.items():
        assert isinstance(k, str)
        assert k in sample_df.columns
        assert isinstance(v, dict)
        assert v == dt.columns[k].semantic_types


def test_check_semantic_types_errors(sample_df):
    error_message = 'semantic_types must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_semantic_types(sample_df, semantic_types='type')

    bad_semantic_types_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("semantic_types contains columns that are not present in dataframe: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_semantic_types(sample_df, bad_semantic_types_keys)


def test_set_logical_types(sample_df):
    semantic_types = {
        'full_name': {'tag1': {}},
        'email': {'tag2': {'option1': 'value1'}},
        'phone_number': {'tag3': {}},
        'signup_date': {'secondary_time_index': {'columns': ['expired']}},
    }
    dt = DataTable(sample_df, semantic_types=semantic_types)

    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['email'].logical_type == NaturalLanguage
    assert dt.columns['phone_number'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == WholeNumber
    assert dt.columns['signup_date'].logical_type == Datetime
    original_name_column = dt.columns['full_name']

    dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
    })

    assert dt.columns['full_name'].logical_type == FullName
    assert dt.columns['email'].logical_type == EmailAddress
    assert dt.columns['phone_number'].logical_type == PhoneNumber
    assert dt.columns['age'].logical_type == Double

    # Verify new column object was created
    new_name_column = dt.columns['full_name']
    assert new_name_column != original_name_column

    # Verify semantic types were not changed
    assert dt.columns['full_name'].semantic_types == semantic_types['full_name']
    assert dt.columns['email'].semantic_types == semantic_types['email']
    assert dt.columns['phone_number'].semantic_types == semantic_types['phone_number']
    assert dt.columns['age'].semantic_types == {}


def test_set_logical_types_invalid_data(sample_df):
    dt = DataTable(sample_df)
    error_message = re.escape("logical_types contains columns that are not present in dataframe: ['birthday']")
    with pytest.raises(LookupError, match=error_message):
        dt.set_logical_types({'birthday': Double})

    error_message = "Invalid logical type specified for 'age'"
    with pytest.raises(TypeError, match=error_message):
        dt.set_logical_types({'age': int})


def test_semantic_types_during_init(sample_df):
    semantic_types = {
        'full_name': 'tag1',
        'email': {'tag2': {'option1': 'value1'}},
        'phone_number': {'tag3': None},
        'signup_date': {'secondary_time_index': {'columns': ['expired']}},
        'age': ['numeric', 'age']
    }
    expected_types = {
        'full_name': {'tag1': {}},
        'email': {'tag2': {'option1': 'value1'}},
        'phone_number': {'tag3': {}},
        'signup_date': {'secondary_time_index': {'columns': ['expired']}},
        'age': {'numeric': {}, 'age': {}}
    }
    dt = DataTable(sample_df, semantic_types=semantic_types)
    assert dt.columns['full_name'].semantic_types == expected_types['full_name']
    assert dt.columns['email'].semantic_types == expected_types['email']
    assert dt.columns['phone_number'].semantic_types == expected_types['phone_number']
    assert dt.columns['signup_date'].semantic_types == expected_types['signup_date']
    assert dt.columns['age'].semantic_types == expected_types['age']


def test_set_semantic_types(sample_df):
    semantic_types = {
        'full_name': 'tag1',
        'age': ['numeric', 'age']
    }
    expected_types = {
        'full_name': {'tag1': {}},
        'age': {'numeric': {}, 'age': {}}
    }
    dt = DataTable(sample_df, semantic_types=semantic_types)
    assert dt.columns['full_name'].semantic_types == expected_types['full_name']
    assert dt.columns['age'].semantic_types == expected_types['age']

    new_types = {
        'full_name': {'new_tag': {'additional': 'value'}},
        'age': 'numeric',
    }
    dt.set_semantic_types(new_types)
    assert dt.columns['full_name'].semantic_types == new_types['full_name']
    assert dt.columns['age'].semantic_types == {'numeric': {}}


def test_replace_none_with_pdna(none_df):
    logical_types = {
        'all_none': NaturalLanguage,
        'some_none': NaturalLanguage,
    }
    dt = DataTable(none_df, logical_types=logical_types, replace_none=True)
    assert dt.df['all_none'].loc[0] is pd.NA
    assert dt.df['all_none'].loc[1] is pd.NA
    assert dt.df['all_none'].loc[2] is pd.NA
    assert dt.df['some_none'].loc[1] is pd.NA


def test_sets_category_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['a', 'b', 'c'], name=column_name),
        pd.Series(['a', None, 'c'], name=column_name),
        pd.Series(['a', np.nan, 'c'], name=column_name),
        pd.Series(['a', pd.NA, 'c'], name=column_name),
    ]

    logical_types = [
        Categorical,
        CountryCode,
        Ordinal,
        SubRegionCode,
        ZIPCode,
    ]

    for series in series_list:
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
            assert dt.columns[column_name].logical_type == logical_type
            assert dt.columns[column_name].dtype == logical_type.pandas_dtype
            assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_string_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['a', 'b', 'c'], name=column_name),
        pd.Series(['a', None, 'c'], name=column_name),
        pd.Series(['a', np.nan, 'c'], name=column_name),
        pd.Series(['a', pd.NA, 'c'], name=column_name),
    ]

    logical_types = [
        FilePath,
        FullName,
        IPAddress,
        LatLong,
        NaturalLanguage,
        PhoneNumber,
        URL,
    ]

    for series in series_list:
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
            assert dt.columns[column_name].logical_type == logical_type
            assert dt.columns[column_name].dtype == logical_type.pandas_dtype
            assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_boolean_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([True, False, True], name=column_name),
        pd.Series([True, None, True], name=column_name),
        pd.Series([True, np.nan, True], name=column_name),
        pd.Series([True, pd.NA, True], name=column_name),
    ]

    logical_type = Boolean
    for series in series_list:
        ltypes = {
            column_name: logical_type,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes, replace_none=False)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_int64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([1, 2, 3], name=column_name),
        pd.Series([1, None, 3], name=column_name),
        pd.Series([1, np.nan, 3], name=column_name),
        pd.Series([1, pd.NA, 3], name=column_name),
    ]

    logical_types = [Integer, WholeNumber]
    for series in series_list:
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes, replace_none=False)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_float64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([1.1, 2, 3], name=column_name),
        pd.Series([1.1, None, 3], name=column_name),
        pd.Series([1.1, np.nan, 3], name=column_name),
        # pd.Series([1.1, pd.NA, 3], name=column_name),  # Fails to cast with pd.NA value
    ]

    logical_type = Double
    for series in series_list:
        ltypes = {
            column_name: logical_type,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes, replace_none=False)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_datetime64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', None, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', np.nan, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', pd.NA, '2020-01-03'], name=column_name),
    ]

    logical_type = Datetime
    for series in series_list:
        ltypes = {
            column_name: logical_type,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes, replace_none=False)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype
