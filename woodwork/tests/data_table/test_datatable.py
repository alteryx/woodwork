import re

import numpy as np
import pandas as pd
import pytest

from woodwork import DataColumn, DataTable
from woodwork.data_table import (
    _check_index,
    _check_logical_types,
    _check_semantic_tags,
    _check_time_index,
    _check_unique_column_names,
    _validate_params
)
from woodwork.logical_types import (
    URL,
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    EmailAddress,
    Filepath,
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
from woodwork.tests.testing_utils import validate_subset_dt


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


def test_datatable_init_with_semantic_tags(sample_df):
    semantic_tags = {
        'id': 'index',
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   semantic_tags=semantic_tags,
                   add_standard_tags=False)

    id_semantic_tags = dt.columns['id'].semantic_tags
    assert isinstance(id_semantic_tags, set)
    assert len(id_semantic_tags) == 1
    assert 'index' in id_semantic_tags


def test_datatable_adds_standard_semantic_tags(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types={
                       'id': Categorical,
                       'age': WholeNumber,
                   })

    assert dt.semantic_tags['id'] == {'category'}
    assert dt.semantic_tags['age'] == {'numeric'}


def test_validate_params_errors(sample_df):
    error_message = 'Dataframe must be a pandas.DataFrame'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=pd.Series(),
                         name=None,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         semantic_tags=None)

    error_message = 'DataTable name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=sample_df,
                         name=1,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         semantic_tags=None)


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
        assert isinstance(tag, set)


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


def test_datatable_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'age': ['numeric', 'age']
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags)
    assert isinstance(dt.semantic_tags, dict)
    assert set(dt.semantic_tags.keys()) == set(sample_df.columns)
    for k, v in dt.semantic_tags.items():
        assert isinstance(k, str)
        assert k in sample_df.columns
        assert isinstance(v, set)
        assert v == dt.columns[k].semantic_tags


def test_check_semantic_tags_errors(sample_df):
    error_message = 'semantic_tags must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_semantic_tags(sample_df, semantic_tags='type')

    bad_semantic_tags_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("semantic_tags contains columns that are not present in dataframe: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_semantic_tags(sample_df, bad_semantic_tags_keys)


def test_set_logical_types(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'phone_number': ['tag3', 'tag2'],
        'signup_date': {'secondary_time_index'},
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags, add_standard_tags=True)
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['email'].logical_type == NaturalLanguage
    assert dt.columns['phone_number'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == WholeNumber
    assert dt.columns['signup_date'].logical_type == Datetime
    original_name_column = dt.columns['full_name']
    original_signup_column = dt.columns['signup_date']

    dt.set_logical_types({
        'full_name': Categorical,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
    })

    assert dt.columns['full_name'].logical_type == Categorical
    assert dt.columns['email'].logical_type == EmailAddress
    assert dt.columns['phone_number'].logical_type == PhoneNumber
    assert dt.columns['age'].logical_type == Double

    # Verify new column object was created
    new_name_column = dt.columns['full_name']
    assert new_name_column is not original_name_column

    # Verify semantic tags were reset to standard tags
    assert dt.columns['full_name'].semantic_tags == {'category'}
    assert dt.columns['email'].semantic_tags == set()
    assert dt.columns['phone_number'].semantic_tags == set()
    assert dt.columns['age'].semantic_tags == {'numeric'}

    # Verify signup date column was unchanged
    assert dt.columns['signup_date'] is original_signup_column
    assert dt.columns['signup_date'].logical_type == Datetime
    assert dt.columns['signup_date'].semantic_tags == {'secondary_time_index'}


def test_set_logical_types_invalid_data(sample_df):
    dt = DataTable(sample_df)
    error_message = re.escape("logical_types contains columns that are not present in dataframe: ['birthday']")
    with pytest.raises(LookupError, match=error_message):
        dt.set_logical_types({'birthday': Double})

    error_message = "Invalid logical type specified for 'age'"
    with pytest.raises(TypeError, match=error_message):
        dt.set_logical_types({'age': int})


def test_semantic_tags_during_init(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'phone_number': ['tag3'],
        'signup_date': ['secondary_time_index'],
        'age': ['numeric', 'age']
    }
    expected_types = {
        'full_name': {'tag1'},
        'email': {'tag2'},
        'phone_number': {'tag3'},
        'signup_date': {'secondary_time_index'},
        'age': {'numeric', 'age'}
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags)
    assert dt.columns['full_name'].semantic_tags == expected_types['full_name']
    assert dt.columns['email'].semantic_tags == expected_types['email']
    assert dt.columns['phone_number'].semantic_tags == expected_types['phone_number']
    assert dt.columns['signup_date'].semantic_tags == expected_types['signup_date']
    assert dt.columns['age'].semantic_tags == expected_types['age']


def test_set_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'age': ['numeric', 'age']
    }
    expected_tags = {
        'full_name': {'tag1'},
        'age': {'numeric', 'age'}
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags)
    assert dt.columns['full_name'].semantic_tags == expected_tags['full_name']
    assert dt.columns['age'].semantic_tags == expected_tags['age']

    new_tags = {
        'full_name': ['new_tag'],
        'age': 'numeric',
    }
    dt.set_semantic_tags(new_tags)
    assert dt.columns['full_name'].semantic_tags == {'new_tag'}
    assert dt.columns['age'].semantic_tags == {'numeric'}


def test_add_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'age': ['numeric', 'age']
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags, add_standard_tags=False)

    new_tags = {
        'full_name': ['list_tag'],
        'age': 'str_tag',
        'id': {'set_tag'}
    }
    dt.add_semantic_tags(new_tags)
    assert dt.columns['full_name'].semantic_tags == {'tag1', 'list_tag'}
    assert dt.columns['age'].semantic_tags == {'numeric', 'age', 'str_tag'}
    assert dt.columns['id'].semantic_tags == {'set_tag'}


def test_reset_all_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'age': 'age'
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags, add_standard_tags=True)

    dt.reset_semantic_tags()
    assert dt.columns['full_name'].semantic_tags == set()
    assert dt.columns['age'].semantic_tags == {'numeric'}


def test_reset_selected_column_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'age': 'age'
    }

    input_types = ['age', ['age'], {'age'}]
    for input_type in input_types:
        dt = DataTable(sample_df, semantic_tags=semantic_tags, add_standard_tags=True)
        dt.reset_semantic_tags(input_type)
        assert dt.columns['full_name'].semantic_tags == {'tag1'}
        assert dt.columns['age'].semantic_tags == {'numeric'}


def test_reset_semantic_tags_invalid_column(sample_df):
    dt = DataTable(sample_df)
    error_msg = "Input contains columns that are not present in dataframe: 'invalid_column'"
    with pytest.raises(LookupError, match=error_msg):
        dt.reset_semantic_tags('invalid_column')


def test_remove_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': ['tag1', 'tag2', 'tag3'],
        'age': ['numeric', 'age'],
        'id': ['tag1', 'tag2']
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags, add_standard_tags=False)
    tags_to_remove = {
        'full_name': ['tag1', 'tag3'],
        'age': 'numeric',
        'id': {'tag1'}
    }
    dt.remove_semantic_tags(tags_to_remove)
    assert dt.columns['full_name'].semantic_tags == {'tag2'}
    assert dt.columns['age'].semantic_tags == {'age'}
    assert dt.columns['id'].semantic_tags == {'tag2'}


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
        pd.Series(['a', pd.NaT, 'c'], name=column_name),
    ]

    logical_types = [
        Categorical,
        CountryCode,
        Ordinal,
        SubRegionCode,
        ZIPCode,
    ]

    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
            assert dt.columns[column_name].logical_type == logical_type
            assert dt.columns[column_name].dtype == logical_type.pandas_dtype
            assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_category_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series(['a', 'b', 'c'], name=column_name)
    series = series.astype('object')
    logical_types = [
        Categorical,
        CountryCode,
        Ordinal,
        SubRegionCode,
        ZIPCode,
    ]

    for logical_type in logical_types:
        ltypes = {
            column_name: NaturalLanguage,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        dt.set_logical_types({column_name: logical_type})
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
        Filepath,
        FullName,
        IPAddress,
        LatLong,
        NaturalLanguage,
        PhoneNumber,
        URL,
    ]

    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
            assert dt.columns[column_name].logical_type == logical_type
            assert dt.columns[column_name].dtype == logical_type.pandas_dtype
            assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_string_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series(['a', 'b', 'c'], name=column_name)
    series = series.astype('object')
    logical_types = [
        Filepath,
        FullName,
        IPAddress,
        LatLong,
        NaturalLanguage,
        PhoneNumber,
        URL,
    ]

    for logical_type in logical_types:
        ltypes = {
            column_name: Categorical,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        dt.set_logical_types({column_name: logical_type})
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
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes, replace_none=False)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_boolean_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series([0, 1, 0], name=column_name)
    series = series.astype('object')
    ltypes = {
        column_name: WholeNumber,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    dt.set_logical_types({column_name: Boolean})
    assert dt.columns[column_name].logical_type == Boolean
    assert dt.columns[column_name].dtype == Boolean.pandas_dtype
    assert dt.dataframe[column_name].dtype == Boolean.pandas_dtype


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
        series = series.astype('object')
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes, replace_none=False)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_int64_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series([1.0, 2.0, 1.0], name=column_name)
    series = series.astype('object')
    logical_types = [
        Integer,
        WholeNumber,
    ]

    for logical_type in logical_types:
        ltypes = {
            column_name: Double,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        dt.set_logical_types({column_name: logical_type})
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_float64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([1.1, 2, 3], name=column_name),
        pd.Series([1.1, None, 3], name=column_name),
        pd.Series([1.1, np.nan, 3], name=column_name),
    ]

    logical_type = Double
    for series in series_list:
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes, replace_none=False)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_float64_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series([0, 1, 0], name=column_name)
    series = series.astype('object')
    ltypes = {
        column_name: WholeNumber,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    dt.set_logical_types({column_name: Double})
    assert dt.columns[column_name].logical_type == Double
    assert dt.columns[column_name].dtype == Double.pandas_dtype
    assert dt.dataframe[column_name].dtype == Double.pandas_dtype


def test_sets_datetime64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', None, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', np.nan, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', pd.NA, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', pd.NaT, '2020-01-03'], name=column_name),
    ]

    logical_type = Datetime
    for series in series_list:
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes, replace_none=False)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.dataframe[column_name].dtype == logical_type.pandas_dtype


def test_sets_datetime_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name)
    series = series.astype('object')
    ltypes = {
        column_name: NaturalLanguage,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    dt.set_logical_types({column_name: Datetime})
    assert dt.columns[column_name].logical_type == Datetime
    assert dt.columns[column_name].dtype == Datetime.pandas_dtype
    assert dt.dataframe[column_name].dtype == Datetime.pandas_dtype


def test_invalid_dtype_casting():
    column_name = 'test_series'

    # Cannot cast a column with pd.NA to Double
    series = pd.Series([1.1, pd.NA, 3], name=column_name)
    ltypes = {
        column_name: Double,
    }
    err_msg = 'Error converting datatype for column test_series from type object to type ' \
        'float64. Please confirm the underlying data is consistent with logical type Double.'
    with pytest.raises(TypeError, match=err_msg):
        DataTable(pd.DataFrame(series), logical_types=ltypes)

    # Cannot cast Datetime to Double
    series = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name)
    ltypes = {
        column_name: Datetime,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    err_msg = 'Error converting datatype for column test_series from type datetime64[ns] to type ' \
        'float64. Please confirm the underlying data is consistent with logical type Double.'
    with pytest.raises(TypeError, match=re.escape(err_msg)):
        dt.set_logical_types({column_name: Double})

    # Cannot cast invalid strings to whole numbers
    series = pd.Series(['1', 'two', '3'], name=column_name)
    ltypes = {
        column_name: WholeNumber,
    }
    err_msg = 'Error converting datatype for column test_series from type object to type ' \
        'Int64. Please confirm the underlying data is consistent with logical type WholeNumber.'
    with pytest.raises(TypeError, match=err_msg):
        DataTable(pd.DataFrame(series), logical_types=ltypes)


def test_invalid_select_ltypes(sample_df):
    dt = DataTable(sample_df)
    dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })

    error_message = "Invalid logical type specified: 1"
    with pytest.raises(TypeError, match=error_message):
        dt.select_ltypes(1)

    error_message = "String test is not a valid logical type"
    with pytest.raises(ValueError, match=error_message):
        dt.select_ltypes('test')

    all_types = LogicalType.__subclasses__()
    dt_all_types = dt.select_ltypes(all_types)
    assert len(dt_all_types.columns) == len(dt.columns)
    assert len(dt_all_types.df.columns) == len(dt.df.columns)

    dt_empty = dt.select_ltypes([])
    assert not dt_empty.columns
    assert len(dt_empty.df.columns) == 0

    # Now that there are no columns, repeat the check with all ltypes
    dt_from_empty = dt_empty.select_ltypes(all_types)
    assert not dt_from_empty.columns
    assert len(dt_from_empty.df.columns) == 0


def test_select_ltypes_strings(sample_df):
    dt = DataTable(sample_df)
    dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })

    dt_multiple_ltypes = dt.select_ltypes(['FullName', 'email_address', 'double', 'Boolean', 'datetime'])
    assert len(dt_multiple_ltypes.columns) == 5
    assert 'phone_number' not in dt_multiple_ltypes.columns
    assert 'id' not in dt_multiple_ltypes.columns

    dt_single_ltype = dt.select_ltypes('full_name')
    assert len(dt_single_ltype.columns) == 1


def test_select_ltypes_objects(sample_df):
    dt = DataTable(sample_df)
    dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })

    dt_multiple_ltypes = dt.select_ltypes([FullName, EmailAddress, Double, Boolean, Datetime])
    assert len(dt_multiple_ltypes.columns) == 5
    assert 'phone_number' not in dt_multiple_ltypes.columns
    assert 'id' not in dt_multiple_ltypes.columns

    dt_single_ltype = dt.select_ltypes(FullName)
    assert len(dt_single_ltype.columns) == 1


def test_select_ltypes_mixed(sample_df):
    dt = DataTable(sample_df)
    dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })

    dt_mixed_ltypes = dt.select_ltypes(['FullName', 'email_address', Double])
    assert len(dt_mixed_ltypes.columns) == 3
    assert 'phone_number' not in dt_mixed_ltypes.columns

    # Selecting for an ltype that isn't present should result in an empty DataTable
    dt_not_present = dt.select_ltypes('url')
    assert not dt_not_present.columns


def test_select_ltypes_table(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id')
    dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })
    dt.set_semantic_tags({
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
    })

    dt_no_indices = dt.select_ltypes('phone_number')
    assert dt_no_indices.index is None
    assert dt_no_indices.time_index is None

    dt_with_indices = dt.select_ltypes(['Datetime', 'WholeNumber'])
    assert dt_with_indices.index == 'id'
    assert dt_with_indices.time_index == 'signup_date'

    dt_values = dt.select_ltypes(['FullName'])
    assert dt_values.name == dt.name
    original_col = dt_values.columns['full_name']
    col = dt.columns['full_name']
    assert col.logical_type == original_col.logical_type
    assert col.series.equals(original_col.series)
    assert col.dtype == original_col.dtype
    assert col.semantic_tags == original_col.semantic_tags


def test_new_dt_from_columns(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })
    dt.set_semantic_tags({
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
    })
    empty_dt = dt._new_dt_from_cols([])
    assert len(empty_dt.columns) == 0

    just_index = dt._new_dt_from_cols(['id'])
    assert just_index.index == dt.index
    assert just_index.time_index is None
    validate_subset_dt(just_index, dt)

    just_time_index = dt._new_dt_from_cols(['signup_date'])
    assert just_time_index.time_index == dt.time_index
    assert just_time_index.index is None
    validate_subset_dt(just_time_index, dt)

    transfer_schema = dt._new_dt_from_cols(['phone_number'])
    assert transfer_schema.index is None
    assert transfer_schema.time_index is None
    validate_subset_dt(transfer_schema, dt)


def test_invalid_select_semantic_tags(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt.set_semantic_tags({
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
    })
    err_msg = 'include parameter must be a string, set or list'
    with pytest.raises(TypeError, match=err_msg):
        dt.select_semantic_tags(1)

    err_msg = 'include parameter must contain only strings'
    with pytest.raises(TypeError, match=err_msg):
        dt.select_semantic_tags(['test', 1])

    dt_empty = dt.select_semantic_tags([])
    assert len(dt_empty.columns) == 0


def test_select_semantic_tags(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt.set_semantic_tags({
        'full_name': 'tag1',
        'id': 'index',
        'email': ['tag2'],
        'age': ['numeric', 'tag2'],
        'phone_number': ['tag3', 'tag2'],
        'is_registered': 'category',
        'signup_date': ['secondary_time_index'],
    })

    dt_one_match = dt.select_semantic_tags('numeric')
    assert len(dt_one_match.columns) == 2
    assert 'age' in dt_one_match.columns
    assert 'id' in dt_one_match.columns

    dt_multiple_matches = dt.select_semantic_tags('tag2')
    assert len(dt_multiple_matches.columns) == 3
    assert 'age' in dt_multiple_matches.columns
    assert 'phone_number' in dt_multiple_matches.columns
    assert 'email' in dt_multiple_matches.columns

    dt_multiple_tags = dt.select_semantic_tags(['index', 'numeric', 'secondary_time_index'])
    assert len(dt_multiple_tags.columns) == 3
    assert 'id' in dt_multiple_tags.columns
    assert 'age' in dt_multiple_tags.columns
    assert 'signup_date' in dt_multiple_tags.columns

    dt_overlapping_tags = dt.select_semantic_tags(['numeric', 'tag2'])
    assert len(dt_overlapping_tags.columns) == 4
    assert 'id' in dt_overlapping_tags.columns
    assert 'age' in dt_overlapping_tags.columns
    assert 'phone_number' in dt_overlapping_tags.columns
    assert 'email' in dt_overlapping_tags.columns

    dt_common_tags = dt.select_semantic_tags(['category', 'numeric'])
    assert len(dt_common_tags.columns) == 3
    assert 'id' in dt_common_tags.columns
    assert 'is_registered' in dt_common_tags.columns
    assert 'age' in dt_common_tags.columns


def test_select_semantic_tags_warning(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt.set_semantic_tags({
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
    })

    warning = "The following semantic tags were not present in your DataTable: ['doesnt_exist']"
    with pytest.warns(UserWarning, match=warning):
        dt_empty = dt.select_semantic_tags(['doesnt_exist'])
    assert len(dt_empty.columns) == 0

    warning = "The following semantic tags were not present in your DataTable: ['doesnt_exist']"
    with pytest.warns(UserWarning, match=warning):
        dt_single = dt.select_semantic_tags(['numeric', 'doesnt_exist'])
    assert len(dt_single.columns) == 2

    warning = "The following semantic tags were not present in your DataTable: ['category', 'doesnt_exist']"
    with pytest.warns(UserWarning, match=warning):
        dt_single = dt.select_semantic_tags(['numeric', 'doesnt_exist', 'category', 'tag2'])
    assert len(dt_single.columns) == 3


def test_getitem(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types={'age': WholeNumber},
                   semantic_tags={'age': 'custom_tag'},
                   add_standard_tags=True)

    data_col = dt['age']
    assert isinstance(data_col, DataColumn)
    assert data_col.logical_type == WholeNumber
    assert data_col.semantic_tags == {'numeric', 'custom_tag'}


def test_getitem_invalid_input(sample_df):
    dt = DataTable(sample_df)

    error_msg = 'Column name must be a string'
    with pytest.raises(KeyError, match=error_msg):
        dt[1]

    error_msg = "Column with name 'invalid_column' not found in DataTable"
    with pytest.raises(KeyError, match=error_msg):
        dt['invalid_column']
