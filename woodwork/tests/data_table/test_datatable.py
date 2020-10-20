import re

import numpy as np
import pandas as pd
import pytest

import woodwork as ww
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
    Timedelta,
    WholeNumber,
    ZIPCode
)
from woodwork.tests.testing_utils import mi_between_cols, validate_subset_dt


def test_datatable_init(sample_df):
    dt = DataTable(sample_df)
    df = dt.to_pandas()

    assert dt.name is None
    assert dt.index is None
    assert dt.time_index is None
    assert isinstance(df, pd.DataFrame)
    assert set(dt.columns.keys()) == set(sample_df.columns)
    assert df is sample_df
    pd.testing.assert_frame_equal(df, sample_df)


def test_datatable_copy_param(sample_df):
    dt_with_copy = DataTable(sample_df, copy_dataframe=True)
    assert sample_df is not dt_with_copy.to_pandas()

    dt_no_copy = DataTable(sample_df)
    assert sample_df is dt_no_copy.to_pandas()


def test_datatable_init_with_name_and_index_vals(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   index='id',
                   time_index='signup_date')

    assert dt.name == 'datatable'
    assert dt.index == 'id'
    assert dt.time_index == 'signup_date'
    assert dt.columns[dt.time_index].logical_type == Datetime


def test_datatable_init_with_valid_string_time_index():
    df = pd.DataFrame({
        'id': [0, 1, 2, 3],
        'times': ['2019-01-01', '2019-01-02', '2019-01-03', pd.NA]
    })
    dt = DataTable(df,
                   name='datatable',
                   index='id',
                   time_index='times')

    assert dt.name == 'datatable'
    assert dt.index == 'id'
    assert dt.time_index == 'times'
    assert dt.columns[dt.time_index].logical_type == Datetime


def test_datatable_with_numeric_datetime_time_index():
    df = pd.DataFrame({'ints': pd.Series([1, 2, 3]),
                       'strs': ['1', '2', '3']})
    dt = DataTable(df, time_index='ints', logical_types={'ints': Datetime})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        DataTable(df, name='datatable', time_index='strs')

    assert dt.time_index == 'ints'
    assert dt.to_pandas()['ints'].dtype == 'datetime64[ns]'


def test_datatable_with_numeric_time_index():
    df = pd.DataFrame({'numeric_datetime_index': [1, 2, 3],
                       'normal_dates': ['2020-01-01', '2020-01-02', '2020-01-03']})

    # Set a numeric time index on init
    dt = DataTable(df, time_index='numeric_datetime_index')
    date_col = dt['numeric_datetime_index']
    assert dt.time_index == 'numeric_datetime_index'
    assert date_col.logical_type == WholeNumber
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    # Specify logical type for time index on init
    dt = DataTable(df, time_index='numeric_datetime_index', logical_types={'numeric_datetime_index': 'Double'})
    date_col = dt['numeric_datetime_index']
    assert dt.time_index == 'numeric_datetime_index'
    assert date_col.logical_type == Double
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    # Change time index to normal datetime time index
    dt = dt.set_time_index('normal_dates')
    date_col = dt['numeric_datetime_index']
    assert dt.time_index == 'normal_dates'
    assert date_col.logical_type == Double
    assert date_col.semantic_tags == {'numeric'}

    # Set numeric time index after init
    dt = DataTable(df, logical_types={'numeric_datetime_index': 'Double'})
    dt = dt.set_time_index('numeric_datetime_index')
    date_col = dt['numeric_datetime_index']
    assert dt.time_index == 'numeric_datetime_index'
    assert date_col.logical_type == Double
    assert date_col.semantic_tags == {'time_index', 'numeric'}


def test_datatable_init_with_invalid_string_time_index():
    df = pd.DataFrame({
        'id': [0, 1, 2],
        'times': ['not_a_datetime', '2019-01-02', '2019-01-03']
    })
    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        DataTable(df, name='datatable', time_index='times')


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
        'age': 'WholeNumber',
        'signup_date': 'Datetime'
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types=logical_types,
                   time_index='signup_date')
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == WholeNumber
    assert dt.time_index == 'signup_date'


def test_datatable_init_with_semantic_tags(sample_df):
    semantic_tags = {
        'id': 'custom_tag',
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   semantic_tags=semantic_tags,
                   use_standard_tags=False)

    id_semantic_tags = dt.columns['id'].semantic_tags
    assert isinstance(id_semantic_tags, set)
    assert len(id_semantic_tags) == 1
    assert 'custom_tag' in id_semantic_tags


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
                         semantic_tags=None,
                         make_index=False)

    error_message = 'DataTable name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=sample_df,
                         name=1,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         semantic_tags=None,
                         make_index=False)


def test_check_index_errors(sample_df):
    error_message = 'Index column name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _check_index(dataframe=sample_df, index=1)

    error_message = 'Specified index column `foo` not found in dataframe. To create a new index column, set make_index to True.'
    with pytest.raises(LookupError, match=error_message):
        _check_index(dataframe=sample_df, index='foo')

    error_message = 'Index column must be unique'
    with pytest.raises(LookupError, match=error_message):
        _check_index(sample_df, index='age')

    error_message = 'When setting make_index to True, the name specified for index cannot match an existing column name'
    with pytest.raises(IndexError, match=error_message):
        _check_index(sample_df, index='id', make_index=True)

    error_message = 'When setting make_index to True, the name for the new index must be specified in the index parameter'
    with pytest.raises(IndexError, match=error_message):
        _check_index(sample_df, index=None, make_index=True)


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
    sample_df['formatted_date'] = pd.Series(["2019~01~01", "2019~01~02", "2019~01~03"])
    ymd_format = Datetime(datetime_format='%Y~%m~%d')
    dt = DataTable(sample_df, logical_types={'formatted_date': ymd_format})
    returned_types = dt.types
    assert isinstance(returned_types, pd.DataFrame)
    assert 'Physical Type' in returned_types.columns
    assert 'Logical Type' in returned_types.columns
    assert 'Semantic Tag(s)' in returned_types.columns
    assert returned_types.shape[1] == 3
    assert len(returned_types.index) == len(sample_df.columns)
    assert all([dc.logical_type in LogicalType.__subclasses__() or isinstance(dc.logical_type, LogicalType) for dc in dt.columns.values()])
    correct_logical_types = {
        'id': WholeNumber,
        'full_name': NaturalLanguage,
        'email': NaturalLanguage,
        'phone_number': NaturalLanguage,
        'age': WholeNumber,
        'signup_date': Datetime,
        'is_registered': Boolean,
        'formatted_date': ymd_format
    }
    correct_logical_types = pd.Series(list(correct_logical_types.values()),
                                      index=list(correct_logical_types.keys()))
    assert correct_logical_types.equals(returned_types['Logical Type'])
    for tag in returned_types['Semantic Tag(s)']:
        assert isinstance(tag, set)


def test_datatable_ltypes(sample_df):
    dt = DataTable(sample_df)
    returned_types = dt.ltypes
    assert isinstance(returned_types, pd.Series)
    assert returned_types.name == 'Logical Type'
    assert len(returned_types.index) == len(sample_df.columns)
    assert all([issubclass(logical_type, LogicalType) for logical_type in returned_types.values])
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
    assert correct_logical_types.equals(returned_types)


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
    dt = DataTable(sample_df, semantic_tags=semantic_tags, use_standard_tags=True)

    new_dt = dt.set_logical_types({
        'full_name': Categorical,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
    })

    # Verify original types were not changed
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['email'].logical_type == NaturalLanguage
    assert dt.columns['phone_number'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == WholeNumber
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
    new_dt = dt.set_semantic_tags(new_tags)
    # Verify original tags were not changed
    assert dt.columns['full_name'].semantic_tags == {'tag1'}
    assert dt.columns['age'].semantic_tags == {'numeric', 'age'}

    assert new_dt is not dt
    assert new_dt.columns['full_name'].semantic_tags == {'new_tag'}
    assert new_dt.columns['age'].semantic_tags == {'numeric'}


def test_set_semantic_tags_with_index(sample_df):
    dt = DataTable(sample_df, index='id', use_standard_tags=False)
    assert dt.columns['id'].semantic_tags == {'index'}

    new_tags = {
        'id': 'new_tag',
    }
    dt = dt.set_semantic_tags(new_tags)
    assert dt.columns['id'].semantic_tags == {'index', 'new_tag'}
    dt = dt.set_semantic_tags(new_tags, retain_index_tags=False)
    assert dt.columns['id'].semantic_tags == {'new_tag'}


def test_set_semantic_tags_with_time_index(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', use_standard_tags=False)
    assert dt.columns['signup_date'].semantic_tags == {'time_index'}

    new_tags = {
        'signup_date': 'new_tag',
    }
    dt = dt.set_semantic_tags(new_tags)
    assert dt.columns['signup_date'].semantic_tags == {'time_index', 'new_tag'}
    dt = dt.set_semantic_tags(new_tags, retain_index_tags=False)
    assert dt.columns['signup_date'].semantic_tags == {'new_tag'}


def test_add_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'age': ['numeric', 'age']
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags, use_standard_tags=False)

    new_tags = {
        'full_name': ['list_tag'],
        'age': 'str_tag',
        'id': {'set_tag'}
    }
    new_dt = dt.add_semantic_tags(new_tags)
    # Verify original tags were not changed
    assert dt.columns['full_name'].semantic_tags == {'tag1'}
    assert dt.columns['age'].semantic_tags == {'numeric', 'age'}

    assert new_dt is not dt
    assert new_dt.columns['full_name'].semantic_tags == {'tag1', 'list_tag'}
    assert new_dt.columns['age'].semantic_tags == {'numeric', 'age', 'str_tag'}
    assert new_dt.columns['id'].semantic_tags == {'set_tag'}


def test_reset_all_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'age': 'age'
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags, use_standard_tags=True)

    new_dt = dt.reset_semantic_tags()
    # Verify original tags were not changed
    assert dt.columns['full_name'].semantic_tags == {'tag1'}
    assert dt.columns['age'].semantic_tags == {'numeric', 'age'}

    assert new_dt is not dt
    assert new_dt.columns['full_name'].semantic_tags == set()
    assert new_dt.columns['age'].semantic_tags == {'numeric'}


def test_reset_selected_column_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'age': 'age'
    }

    input_types = ['age', ['age'], {'age'}]
    for input_type in input_types:
        dt = DataTable(sample_df, semantic_tags=semantic_tags, use_standard_tags=True)
        dt = dt.reset_semantic_tags(input_type)
        assert dt.columns['full_name'].semantic_tags == {'tag1'}
        assert dt.columns['age'].semantic_tags == {'numeric'}


def test_reset_semantic_tags_with_index(sample_df):
    semantic_tags = {
        'id': 'tag1',
    }
    dt = DataTable(sample_df,
                   index='id',
                   semantic_tags=semantic_tags,
                   use_standard_tags=False)
    assert dt['id'].semantic_tags == {'index', 'tag1'}
    dt = dt.reset_semantic_tags('id', retain_index_tags=True)
    assert dt['id'].semantic_tags == {'index'}
    dt = dt.reset_semantic_tags('id')
    assert dt['id'].semantic_tags == set()


def test_reset_semantic_tags_with_time_index(sample_df):
    semantic_tags = {
        'signup_date': 'tag1',
    }
    dt = DataTable(sample_df,
                   time_index='signup_date',
                   semantic_tags=semantic_tags,
                   use_standard_tags=False)
    assert dt['signup_date'].semantic_tags == {'time_index', 'tag1'}
    dt = dt.reset_semantic_tags('signup_date', retain_index_tags=True)
    assert dt['signup_date'].semantic_tags == {'time_index'}
    dt = dt.reset_semantic_tags('signup_date')
    assert dt['signup_date'].semantic_tags == set()


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
    dt = DataTable(sample_df, semantic_tags=semantic_tags, use_standard_tags=False)
    tags_to_remove = {
        'full_name': ['tag1', 'tag3'],
        'age': 'numeric',
        'id': {'tag1'}
    }
    new_dt = dt.remove_semantic_tags(tags_to_remove)
    # Verify original tags were not changed
    assert dt.columns['full_name'].semantic_tags == {'tag1', 'tag2', 'tag3'}
    assert dt.columns['age'].semantic_tags == {'numeric', 'age'}
    assert dt.columns['id'].semantic_tags == {'tag1', 'tag2'}

    assert new_dt is not dt
    assert new_dt.columns['full_name'].semantic_tags == {'tag2'}
    assert new_dt.columns['age'].semantic_tags == {'age'}
    assert new_dt.columns['id'].semantic_tags == {'tag2'}


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
        Ordinal(order=['a', 'b', 'c']),
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
            assert dt.to_pandas()[column_name].dtype == logical_type.pandas_dtype


def test_sets_category_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series(['a', 'b', 'c'], name=column_name)
    series = series.astype('object')
    logical_types = [
        Categorical,
        CountryCode,
        Ordinal(order=['a', 'b', 'c']),
        SubRegionCode,
        ZIPCode,
    ]

    for logical_type in logical_types:
        ltypes = {
            column_name: NaturalLanguage,
        }
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        dt = dt.set_logical_types({column_name: logical_type})
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_pandas()[column_name].dtype == logical_type.pandas_dtype


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
            assert dt.to_pandas()[column_name].dtype == logical_type.pandas_dtype


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
        dt = dt.set_logical_types({column_name: logical_type})
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_pandas()[column_name].dtype == logical_type.pandas_dtype


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
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_pandas()[column_name].dtype == logical_type.pandas_dtype


def test_sets_boolean_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series([0, 1, 0], name=column_name)
    series = series.astype('object')
    ltypes = {
        column_name: WholeNumber,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    dt = dt.set_logical_types({column_name: Boolean})
    assert dt.columns[column_name].logical_type == Boolean
    assert dt.columns[column_name].dtype == Boolean.pandas_dtype
    assert dt.to_pandas()[column_name].dtype == Boolean.pandas_dtype


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
            dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_pandas()[column_name].dtype == logical_type.pandas_dtype


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
        dt = dt.set_logical_types({column_name: logical_type})
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_pandas()[column_name].dtype == logical_type.pandas_dtype


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
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_pandas()[column_name].dtype == logical_type.pandas_dtype


def test_sets_float64_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series([0, 1, 0], name=column_name)
    series = series.astype('object')
    ltypes = {
        column_name: WholeNumber,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    dt = dt.set_logical_types({column_name: Double})
    assert dt.columns[column_name].logical_type == Double
    assert dt.columns[column_name].dtype == Double.pandas_dtype
    assert dt.to_pandas()[column_name].dtype == Double.pandas_dtype


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
        dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
        assert dt.columns[column_name].logical_type == logical_type
        assert dt.columns[column_name].dtype == logical_type.pandas_dtype
        assert dt.to_pandas()[column_name].dtype == logical_type.pandas_dtype


def test_sets_datetime_dtype_on_update():
    column_name = 'test_series'
    series = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name)
    series = series.astype('object')
    ltypes = {
        column_name: NaturalLanguage,
    }
    dt = DataTable(pd.DataFrame(series), logical_types=ltypes)
    dt = dt.set_logical_types({column_name: Datetime})
    assert dt.columns[column_name].logical_type == Datetime
    assert dt.columns[column_name].dtype == Datetime.pandas_dtype
    assert dt.to_pandas()[column_name].dtype == Datetime.pandas_dtype


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


def test_int_dtype_inference_on_init():
    df = pd.DataFrame({
        'ints_no_nans': pd.Series([1, 2]),
        'ints_nan': pd.Series([1, np.nan]),
        'ints_NA': pd.Series([1, pd.NA]),
        'ints_NA_specified': pd.Series([1, pd.NA], dtype='Int64')})
    df_from_dt = DataTable(df).to_pandas()

    assert df_from_dt['ints_no_nans'].dtype == 'Int64'
    assert df_from_dt['ints_nan'].dtype == 'float64'
    assert df_from_dt['ints_NA'].dtype == 'category'
    assert df_from_dt['ints_NA_specified'].dtype == 'Int64'


def test_bool_dtype_inference_on_init():
    df = pd.DataFrame({
        'bools_no_nans': pd.Series([True, False]),
        'bool_nan': pd.Series([True, np.nan]),
        'bool_NA': pd.Series([True, pd.NA]),
        'bool_NA_specified': pd.Series([True, pd.NA], dtype="boolean")})
    df_from_dt = DataTable(df).to_pandas()

    assert df_from_dt['bools_no_nans'].dtype == 'boolean'
    assert df_from_dt['bool_nan'].dtype == 'category'
    assert df_from_dt['bool_NA'].dtype == 'category'
    assert df_from_dt['bool_NA_specified'].dtype == 'boolean'


def test_str_dtype_inference_on_init():
    df = pd.DataFrame({
        'str_no_nans': pd.Series(['a', 'b']),
        'str_nan': pd.Series(['a', np.nan]),
        'str_NA': pd.Series(['a', pd.NA]),
        'str_NA_specified': pd.Series([1, pd.NA], dtype="string"),
        'long_str_NA_specified': pd.Series(['this is a very long sentence inferred as a string', pd.NA], dtype="string"),
        'long_str_NA': pd.Series(['this is a very long sentence inferred as a string', pd.NA])
    })
    df_from_dt = DataTable(df).to_pandas()

    assert df_from_dt['str_no_nans'].dtype == 'category'
    assert df_from_dt['str_nan'].dtype == 'category'
    assert df_from_dt['str_NA'].dtype == 'category'
    assert df_from_dt['str_NA_specified'].dtype == 'category'
    assert df_from_dt['long_str_NA_specified'].dtype == 'string'
    assert df_from_dt['long_str_NA'].dtype == 'string'


def test_float_dtype_inference_on_init():
    df = pd.DataFrame({
        'floats_no_nans': pd.Series([1.1, 2.2]),
        'floats_nan': pd.Series([1.1, np.nan]),
        'floats_NA': pd.Series([1.1, pd.NA]),
        'floats_nan_specified': pd.Series([1.1, np.nan], dtype='float')})
    df_from_dt = DataTable(df).to_pandas()

    assert df_from_dt['floats_no_nans'].dtype == 'float64'
    assert df_from_dt['floats_nan'].dtype == 'float64'
    assert df_from_dt['floats_NA'].dtype == 'category'
    assert df_from_dt['floats_nan_specified'].dtype == 'float64'


def test_datetime_dtype_inference_on_init():
    df = pd.DataFrame({
        'date_no_nans': pd.Series([pd.to_datetime('2020-09-01')] * 2),
        'date_nan': pd.Series([pd.to_datetime('2020-09-01'), np.nan]),
        'date_NA': pd.Series([pd.to_datetime('2020-09-01'), pd.NA]),
        'date_NaT': pd.Series([pd.to_datetime('2020-09-01'), pd.NaT]),
        'date_NA_specified': pd.Series([pd.to_datetime('2020-09-01'), pd.NA], dtype='datetime64[ns]')})
    df_from_dt = DataTable(df).to_pandas()

    assert df_from_dt['date_no_nans'].dtype == 'datetime64[ns]'
    assert df_from_dt['date_nan'].dtype == 'datetime64[ns]'
    assert df_from_dt['date_NA'].dtype == 'datetime64[ns]'
    assert df_from_dt['date_NaT'].dtype == 'datetime64[ns]'
    assert df_from_dt['date_NA_specified'].dtype == 'datetime64[ns]'


def test_timedelta_dtype_inference_on_init():
    df = pd.DataFrame({
        'delta_no_nans': (pd.Series([pd.to_datetime('2020-09-01')] * 2) - pd.to_datetime('2020-07-01')),
        'delta_nan': (pd.Series([pd.to_datetime('2020-09-01'), np.nan]) - pd.to_datetime('2020-07-01')),
        'delta_NaT': (pd.Series([pd.to_datetime('2020-09-01'), pd.NaT]) - pd.to_datetime('2020-07-01')),
        'delta_NA_specified': (pd.Series([pd.to_datetime('2020-09-01'), pd.NA], dtype='datetime64[ns]') - pd.to_datetime('2020-07-01')),
    })
    df_from_dt = DataTable(df).to_pandas()

    assert df_from_dt['delta_no_nans'].dtype == 'timedelta64[ns]'
    assert df_from_dt['delta_nan'].dtype == 'timedelta64[ns]'
    assert df_from_dt['delta_NaT'].dtype == 'timedelta64[ns]'
    assert df_from_dt['delta_NA_specified'].dtype == 'timedelta64[ns]'


def test_select_ltypes_warning(sample_df):
    dt = DataTable(sample_df)
    dt = dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })

    warning = 'The following selectors were not present in your DataTable: ZIPCode'
    with pytest.warns(UserWarning, match=warning):
        dt_empty = dt.select(ZIPCode)
    assert len(dt_empty.columns) == 0

    warning = 'The following selectors were not present in your DataTable: ZIPCode'
    with pytest.warns(UserWarning, match=warning):
        dt_empty = dt.select(['ZIPCode', PhoneNumber])
    assert len(dt_empty.columns) == 1

    all_types = LogicalType.__subclasses__()
    warning = 'The following selectors were not present in your DataTable: Categorical, CountryCode, Filepath, IPAddress, Integer, LatLong, NaturalLanguage, Ordinal, SubRegionCode, Timedelta, URL, ZIPCode'
    with pytest.warns(UserWarning, match=warning):
        dt_all_types = dt.select(all_types)
    assert len(dt_all_types.columns) == len(dt.columns)
    assert len(dt_all_types.to_pandas().columns) == len(dt.to_pandas().columns)


def test_select_ltypes_strings(sample_df):
    dt = DataTable(sample_df)
    dt = dt.set_logical_types({
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
    dt = dt.set_logical_types({
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
    dt = dt.set_logical_types({
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
    dt = dt.set_logical_types({
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

    dt_no_indices = dt.select('phone_number')
    assert dt_no_indices.index is None
    assert dt_no_indices.time_index is None

    dt_with_indices = dt.select(['Datetime', 'WholeNumber'])
    assert dt_with_indices.index == 'id'
    assert dt_with_indices.time_index == 'signup_date'

    dt_values = dt.select(['FullName'])
    assert dt_values.name == dt.name
    original_col = dt_values.columns['full_name']
    col = dt.columns['full_name']
    assert col.logical_type == original_col.logical_type
    assert col.to_pandas().equals(original_col.to_pandas())
    assert col.dtype == original_col.dtype
    assert col.semantic_tags == original_col.semantic_tags


def test_new_dt_from_columns(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt = dt.set_logical_types({
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


def test_select_semantic_tags(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', name='dt_name')
    dt = dt.set_semantic_tags({
        'full_name': 'tag1',
        'email': ['tag2'],
        'age': ['numeric', 'tag2'],
        'phone_number': ['tag3', 'tag2'],
        'is_registered': 'category',
    })

    dt_one_match = dt.select('numeric')
    assert len(dt_one_match.columns) == 2
    assert 'age' in dt_one_match.columns
    assert 'id' in dt_one_match.columns

    dt_multiple_matches = dt.select('tag2')
    assert len(dt_multiple_matches.columns) == 3
    assert 'age' in dt_multiple_matches.columns
    assert 'phone_number' in dt_multiple_matches.columns
    assert 'email' in dt_multiple_matches.columns

    dt_multiple_tags = dt.select(['numeric', 'time_index'])
    assert len(dt_multiple_tags.columns) == 3
    assert 'id' in dt_multiple_tags.columns
    assert 'age' in dt_multiple_tags.columns
    assert 'signup_date' in dt_multiple_tags.columns

    dt_overlapping_tags = dt.select(['numeric', 'tag2'])
    assert len(dt_overlapping_tags.columns) == 4
    assert 'id' in dt_overlapping_tags.columns
    assert 'age' in dt_overlapping_tags.columns
    assert 'phone_number' in dt_overlapping_tags.columns
    assert 'email' in dt_overlapping_tags.columns

    dt_common_tags = dt.select(['category', 'numeric'])
    assert len(dt_common_tags.columns) == 3
    assert 'id' in dt_common_tags.columns
    assert 'is_registered' in dt_common_tags.columns
    assert 'age' in dt_common_tags.columns


def test_select_semantic_tags_warning(sample_df):
    dt = DataTable(sample_df, name='dt_name')
    dt = dt.set_semantic_tags({
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
    })

    warning = "The following selectors were not present in your DataTable: doesnt_exist"
    with pytest.warns(UserWarning, match=warning):
        dt_empty = dt.select(['doesnt_exist'])
    assert len(dt_empty.columns) == 0

    warning = "The following selectors were not present in your DataTable: doesnt_exist"
    with pytest.warns(UserWarning, match=warning):
        dt_single = dt.select(['numeric', 'doesnt_exist'])
    assert len(dt_single.columns) == 2

    warning = "The following selectors were not present in your DataTable: category, doesnt_exist"
    with pytest.warns(UserWarning, match=warning):
        dt_single = dt.select(['numeric', 'doesnt_exist', 'category', 'tag2'])
    assert len(dt_single.columns) == 3


def test_pop(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types={'age': WholeNumber},
                   semantic_tags={'age': 'custom_tag'},
                   use_standard_tags=True)
    datacol = dt.pop('age')
    assert isinstance(datacol, DataColumn)
    assert 'custom_tag' in datacol.semantic_tags
    assert datacol.to_pandas().values == [33, 25, 33]
    assert datacol.logical_type == WholeNumber

    assert 'age' not in dt.to_pandas().columns
    assert 'age' not in dt.columns

    assert 'age' not in dt.logical_types.keys()
    assert 'age' not in dt.semantic_tags.keys()


def test_pop_index(sample_df):
    dt = DataTable(sample_df, index='id', name='dt_name')
    assert dt.index == 'id'
    id_col = dt.pop('id')
    assert dt.index is None
    assert 'index' in id_col.semantic_tags


def test_pop_error(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types={'age': WholeNumber},
                   semantic_tags={'age': 'custom_tag'},
                   use_standard_tags=True)

    with pytest.raises(KeyError, match="Column with name \'missing\' not found in DataTable"):
        dt.pop("missing")


def test_getitem(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types={'age': WholeNumber},
                   semantic_tags={'age': 'custom_tag'},
                   use_standard_tags=True)

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


def test_datatable_getitem_list_input(sample_df):
    # Test regular columns
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    df = dt.to_pandas()
    columns = ['age', 'full_name']
    new_dt = dt[columns]
    assert new_dt is not dt
    assert new_dt.to_pandas() is not df
    pd.testing.assert_frame_equal(df[columns], new_dt.to_pandas())
    assert all(new_dt.to_pandas().columns == ['age', 'full_name'])
    assert set(new_dt.columns.keys()) == {'age', 'full_name'}
    assert new_dt.index is None
    assert new_dt.time_index is None

    # Test with index
    columns = ['id', 'full_name']
    new_dt = dt[columns]
    assert new_dt is not dt
    assert new_dt.to_pandas() is not df
    pd.testing.assert_frame_equal(df[columns], new_dt.to_pandas())
    assert all(new_dt.to_pandas().columns == ['id', 'full_name'])
    assert set(new_dt.columns.keys()) == {'id', 'full_name'}
    assert new_dt.index == 'id'
    assert new_dt.time_index is None

    # Test with time_index
    columns = ['id', 'signup_date', 'full_name']
    new_dt = dt[columns]
    assert new_dt is not dt
    assert new_dt.to_pandas() is not df
    pd.testing.assert_frame_equal(df[columns], new_dt.to_pandas())
    assert all(new_dt.to_pandas().columns == ['id', 'signup_date', 'full_name'])
    assert set(new_dt.columns.keys()) == {'id', 'signup_date', 'full_name'}
    assert new_dt.index == 'id'
    assert new_dt.time_index == 'signup_date'

    # Test with empty list selector
    columns = []
    new_dt = dt[columns]
    assert new_dt is not dt
    assert new_dt.to_pandas() is not df
    pd.testing.assert_frame_equal(df[columns], new_dt.to_pandas())
    assert len(new_dt.to_pandas().columns) == 0
    assert set(new_dt.columns.keys()) == set()
    assert new_dt.index is None
    assert new_dt.time_index is None


def test_datatable_getitem_list_warnings(sample_df):
    # Test regular columns
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    columns = ['age', 'invalid_col1', 'invalid_col2']
    error_msg = re.escape("Column(s) 'invalid_col1, invalid_col2' not found in DataTable")
    with pytest.raises(KeyError, match=error_msg):
        dt[columns]

    columns = [1]
    error_msg = 'Column names must be strings'
    with pytest.raises(KeyError, match=error_msg):
        dt[columns]


def test_setitem_invalid_input(sample_df):
    dt = DataTable(sample_df, index='id', time_index='signup_date')

    error_msg = 'Column name must be a string'
    with pytest.raises(KeyError, match=error_msg):
        dt[1] = DataColumn(pd.Series([1, 2, 3], dtype='Int64'),
                           use_standard_tags=False)

    error_msg = 'New column must be of DataColumn type'
    with pytest.raises(ValueError, match=error_msg):
        dt['test'] = pd.Series([1, 2, 3], dtype='Int64')

    error_msg = 'Cannot reassign index. Change column name and then use dt.set_index to reassign index.'
    with pytest.raises(KeyError, match=error_msg):
        dt['id'] = DataColumn(pd.Series([True, False, False]))

    error_msg = 'Cannot reassign time index. Change column name and then use dt.set_time_index to reassign time index.'
    with pytest.raises(KeyError, match=error_msg):
        dt['signup_date'] = DataColumn(pd.Series(['test text', 'file', 'False']))


def test_setitem_different_name(sample_df):
    dt = DataTable(sample_df)

    warning = 'Key, id, does not match the name of the provided DataColumn, wrong.'\
        ' Changing DataColumn name to: id'
    with pytest.warns(UserWarning, match=warning):
        dt['id'] = DataColumn(pd.Series([1, 2, 3], dtype='Int64', name='wrong'),
                              use_standard_tags=False)

    assert dt['id'].name == 'id'
    assert dt['id'].to_pandas().name == 'id'
    assert dt.to_pandas()['id'].name == 'id'
    assert 'wrong' not in dt.columns

    warning = 'Key, new_col, does not match the name of the provided DataColumn, wrong2.'\
        ' Changing DataColumn name to: new_col'
    with pytest.warns(UserWarning, match=warning):
        dt['new_col'] = DataColumn(pd.Series([1, 2, 3], dtype='Int64', name='wrong2'),
                                   use_standard_tags=False)
    assert dt['new_col'].name == 'new_col'
    assert dt['new_col'].to_pandas().name == 'new_col'
    assert dt.to_pandas()['new_col'].name == 'new_col'
    assert 'wrong2' not in dt.columns


def test_setitem_new_column(sample_df):
    dt = DataTable(sample_df)

    new_col = DataColumn(pd.Series([1, 2, 3], dtype='Int64'),
                         use_standard_tags=False)
    dt['test_col2'] = new_col
    updated_df = dt.to_pandas()
    assert 'test_col2' in dt.columns
    assert dt['test_col2'].logical_type == WholeNumber
    assert dt['test_col2'].semantic_tags == set()
    assert 'test_col2' in updated_df.columns
    assert updated_df['test_col2'].dtype == 'Int64'

    # Standard tags and no logical type
    new_col = DataColumn(pd.Series(['new', 'column', 'inserted'], dtype='string'),
                         use_standard_tags=True)
    dt['test_col'] = new_col
    updated_df = dt.to_pandas()
    assert 'test_col' in dt.columns
    assert dt['test_col'].logical_type == Categorical
    assert dt['test_col'].semantic_tags == {'category'}
    assert 'test_col' in updated_df.columns
    assert updated_df['test_col'].dtype == 'category'

    # Add with logical type and semantic tag
    new_col = DataColumn(pd.Series([1, 2, 3]),
                         logical_type=Double,
                         use_standard_tags=False,
                         semantic_tags={'test_tag'})
    dt['test_col3'] = new_col
    updated_df = dt.to_pandas()
    assert 'test_col3' in dt.columns
    assert dt['test_col3'].logical_type == Double
    assert dt['test_col3'].semantic_tags == {'test_tag'}
    assert 'test_col3' in updated_df.columns
    assert updated_df['test_col3'].dtype == 'float'


def test_setitem_overwrite_column(sample_df):
    dt = DataTable(sample_df, index='id',
                   time_index='signup_date',
                   use_standard_tags=True)

    # Change to column no change in types
    original_col = dt['age']
    overwrite_col = DataColumn(pd.Series([1, 2, 3], dtype='Int64'),
                               use_standard_tags=True)
    dt['age'] = overwrite_col
    updated_df = dt.to_pandas()

    assert 'age' in dt.columns
    assert dt['age'].logical_type == original_col.logical_type
    assert dt['age'].semantic_tags == original_col.semantic_tags
    assert 'age' in updated_df.columns
    assert updated_df['age'].dtype == 'Int64'
    assert original_col.to_pandas() is not dt['age'].to_pandas()

    # Change dtype, logical types, and tags with conflicting use_standard_tags
    original_col = dt['full_name']
    overwrite_col = DataColumn(pd.Series([True, False, False], dtype='boolean'),
                               use_standard_tags=False,
                               semantic_tags='test_tag')
    dt['full_name'] = overwrite_col
    updated_df = dt.to_pandas()

    assert 'full_name' in dt.columns
    assert dt['full_name'].logical_type == Boolean
    assert dt['full_name'].semantic_tags == {'test_tag'}
    assert 'full_name' in updated_df.columns
    assert updated_df['full_name'].dtype == 'boolean'
    assert original_col.to_pandas() is not dt['full_name'].to_pandas()


def test_set_index(sample_df):
    # Test setting index with set_index()
    dt = DataTable(sample_df)
    new_dt = dt.set_index('id')
    assert new_dt is not dt
    assert new_dt.index == 'id'
    assert dt.index is None
    assert new_dt.columns['id'].semantic_tags == {'index'}
    non_index_cols = [col for col in new_dt.columns.values() if col.name != 'id']
    assert all(['index' not in col.semantic_tags for col in non_index_cols])
    # Test changing index with set_index()
    new_dt2 = new_dt.set_index('full_name')
    assert new_dt.index == 'id'
    assert new_dt2.columns['full_name'].semantic_tags == {'index'}
    non_index_cols = [col for col in new_dt2.columns.values() if col.name != 'full_name']
    assert all(['index' not in col.semantic_tags for col in non_index_cols])

    # Test setting index using setter
    dt = DataTable(sample_df)
    dt.index = 'id'
    assert dt.index == 'id'
    assert 'index' in dt.columns['id'].semantic_tags
    non_index_cols = [col for col in dt.columns.values() if col.name != 'id']
    assert all(['index' not in col.semantic_tags for col in non_index_cols])
    # Test changing index with setter
    dt.index = 'full_name'
    assert 'index' in dt.columns['full_name'].semantic_tags
    non_index_cols = [col for col in dt.columns.values() if col.name != 'full_name']
    assert all(['index' not in col.semantic_tags for col in non_index_cols])


def test_set_time_index(sample_df):
    # Test setting time index with set_time_index()
    dt = DataTable(sample_df)
    new_dt = dt.set_time_index('signup_date')
    assert new_dt is not dt
    assert dt.time_index is None
    assert new_dt.time_index == 'signup_date'
    assert 'time_index' in new_dt.columns['signup_date'].semantic_tags
    non_index_cols = [col for col in new_dt.columns.values() if col.name != 'signup_date']
    assert all(['time_index' not in col.semantic_tags for col in non_index_cols])

    # Test changing time index with set_time_index()
    sample_df['transaction_date'] = pd.to_datetime('2015-09-02')
    dt = DataTable(sample_df)
    new_dt = dt.set_time_index('signup_date')
    assert new_dt.time_index == 'signup_date'
    new_dt2 = new_dt.set_time_index('transaction_date')
    assert 'time_index' in new_dt2.columns['transaction_date'].semantic_tags
    non_index_cols = [col for col in new_dt2.columns.values() if col.name != 'transaction_date']
    assert all(['time_index' not in col.semantic_tags for col in non_index_cols])

    # Test setting index using setter
    dt = DataTable(sample_df)
    assert dt.time_index is None
    dt.time_index = 'signup_date'
    assert dt.time_index == 'signup_date'
    assert 'time_index' in dt.columns['signup_date'].semantic_tags
    non_index_cols = [col for col in dt.columns.values() if col.name != 'signup_date']
    assert all(['time_index' not in col.semantic_tags for col in non_index_cols])

    # Test changing time index with setter
    sample_df['transaction_date'] = pd.to_datetime('2015-09-02')
    dt = DataTable(sample_df)
    dt.time_index = 'signup_date'
    assert dt.time_index == 'signup_date'
    dt.time_index = 'transaction_date'
    assert 'time_index' in dt.columns['transaction_date'].semantic_tags
    non_index_cols = [col for col in dt.columns.values() if col.name != 'transaction_date']
    assert all(['time_index' not in col.semantic_tags for col in non_index_cols])


def test_datatable_clear_index(sample_df):
    # Test by removing index tag
    dt = DataTable(sample_df, index='id')
    assert dt.index == 'id'
    dt = dt.remove_semantic_tags({'id': 'index'})
    assert dt.index is None
    assert all(['index' not in col.semantic_tags for col in dt.columns.values()])

    # Test using setter
    dt = DataTable(sample_df, index='id')
    assert dt.index == 'id'
    dt.index = None
    assert dt.index is None
    assert all(['index' not in col.semantic_tags for col in dt.columns.values()])


def test_datatable_clear_time_index(sample_df):
    # Test by removing time_index tag
    dt = DataTable(sample_df, time_index='signup_date')
    assert dt.time_index == 'signup_date'
    dt = dt.remove_semantic_tags({'signup_date': 'time_index'})
    assert dt.time_index is None
    assert all(['time_index' not in col.semantic_tags for col in dt.columns.values()])

    # Test using setter
    dt = DataTable(sample_df, time_index='signup_date')
    assert dt.time_index == 'signup_date'
    dt.time_index = None
    assert dt.time_index is None
    assert all(['time_index' not in col.semantic_tags for col in dt.columns.values()])


def test_select_invalid_inputs(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt = dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'age': Double,
        'signup_date': Datetime,
    })
    dt = dt.set_semantic_tags({
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
    })

    err_msg = "Invalid selector used in include: 1 must be either a string or LogicalType"
    with pytest.raises(TypeError, match=err_msg):
        dt.select(['boolean', 'index', Double, 1])

    dt_empty = dt.select([])
    assert len(dt_empty.columns) == 0


def test_select_single_inputs(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt = dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'signup_date': Datetime(datetime_format='%Y-%m-%d')
    })
    dt = dt.set_semantic_tags({
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
        'signup_date': 'date_of_birth'
    })

    dt_ltype_string = dt.select('full_name')
    assert len(dt_ltype_string.columns) == 1
    assert 'full_name' in dt_ltype_string.columns

    dt_ltype_obj = dt.select(WholeNumber)
    assert len(dt_ltype_obj.columns) == 2
    assert 'age' in dt_ltype_obj.columns
    assert 'id' in dt_ltype_obj.columns

    dt_tag_string = dt.select('index')
    assert len(dt_tag_string.columns) == 1
    assert 'id' in dt_tag_string.columns

    dt_tag_instantiated = dt.select('Datetime')
    assert len(dt_tag_instantiated.columns) == 1
    assert 'signup_date' in dt_tag_instantiated.columns


def test_select_list_inputs(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt = dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'signup_date': Datetime(datetime_format='%Y-%m-%d'),
    })
    dt = dt.set_semantic_tags({
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
        'signup_date': 'date_of_birth',
        'email': 'tag2',
        'is_registered': 'category'
    })

    dt_just_strings = dt.select(['FullName', 'index', 'tag2', 'boolean'])
    assert len(dt_just_strings.columns) == 4
    assert 'id' in dt_just_strings.columns
    assert 'full_name' in dt_just_strings.columns
    assert 'email' in dt_just_strings.columns
    assert 'is_registered' in dt_just_strings.columns

    dt_mixed_selectors = dt.select([FullName, 'index', 'time_index', WholeNumber])
    assert len(dt_mixed_selectors.columns) == 4
    assert 'id' in dt_mixed_selectors.columns
    assert 'full_name' in dt_mixed_selectors.columns
    assert 'signup_date' in dt_mixed_selectors.columns
    assert 'age' in dt_mixed_selectors.columns

    dt_common_tags = dt.select(['category', 'numeric', Boolean, Datetime])
    assert len(dt_common_tags.columns) == 3
    assert 'is_registered' in dt_common_tags.columns
    assert 'age' in dt_common_tags.columns
    assert 'signup_date' in dt_common_tags.columns


def test_select_warnings(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt = dt.set_logical_types({
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'signup_date': Datetime(datetime_format='%Y-%m-%d'),
    })
    dt = dt.set_semantic_tags({
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
        'signup_date': 'date_of_birth',
        'email': 'tag2'
    })

    warning = 'The following selectors were not present in your DataTable: doesnt_exist'
    with pytest.warns(UserWarning, match=warning):
        dt_empty = dt.select(['doesnt_exist'])
    assert len(dt_empty.columns) == 0

    warning = 'The following selectors were not present in your DataTable: category, doesnt_exist'
    with pytest.warns(UserWarning, match=warning):
        dt_multiple_unused = dt.select(['doesnt_exist', 'boolean', 'category', PhoneNumber])
    assert len(dt_multiple_unused.columns) == 2

    warning = 'The following selectors were not present in your DataTable: ZIPCode, doesnt_exist'
    with pytest.warns(UserWarning, match=warning):
        dt_unused_ltype = dt.select(['date_of_birth', 'doesnt_exist', ZIPCode, WholeNumber])
    assert len(dt_unused_ltype.columns) == 3


def test_select_instantiated():
    ymd_format = Datetime(datetime_format='%Y~%m~%d')

    df = pd.DataFrame({
        'dates': ["2019/01/01", "2019/01/02", "2019/01/03"],
        'ymd': ["2019~01~01", "2019~01~02", "2019~01~03"],
    })
    dt = DataTable(df,
                   logical_types={'ymd': ymd_format,
                                  'dates': Datetime})

    dt = dt.select('Datetime')
    assert len(dt.columns) == 2

    err_msg = "Invalid selector used in include: Datetime cannot be instantiated"
    with pytest.raises(TypeError, match=err_msg):
        dt.select(ymd_format)


def test_filter_cols(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')

    filtered = dt._filter_cols(include='email', col_names=True)
    assert filtered == ['email']

    filtered_log_type_string = dt._filter_cols(include='NaturalLanguage')
    filtered_log_type = dt._filter_cols(include=NaturalLanguage)
    assert filtered_log_type == filtered_log_type_string

    filtered_semantic_tag = dt._filter_cols(include='numeric')
    assert filtered_semantic_tag == ['age']

    filtered_multiple = dt._filter_cols(include=['numeric'])
    expected = ['phone_number', 'age']
    for col in filtered_multiple:
        assert col in expected

    filtered_multiple_overlap = dt._filter_cols(include=['NaturalLanguage', 'email'], col_names=True)
    expected = ['full_name', 'phone_number', 'email']
    for col in filtered_multiple_overlap:
        assert col in expected


def test_filter_cols_errors(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')

    with pytest.warns(UserWarning, match='The following selectors were not present in your DataTable: nothing'):
        filter_no_matches = dt._filter_cols(include='nothing')
    assert filter_no_matches == []


def test_datetime_inference_with_format_param():
    df = pd.DataFrame({
        'index': [0, 1, 2],
        'dates': ["2019/01/01", "2019/01/02", "2019/01/03"],
        'ymd_special': ["2019~01~01", "2019~01~02", "2019~01~03"],
        'mdy_special': pd.Series(['3~11~2000', '3~12~2000', '3~13~2000'], dtype='string'),
    })
    dt = DataTable(df,
                   name='dt_name',
                   logical_types={'ymd_special': Datetime(datetime_format='%Y~%m~%d'),
                                  'mdy_special': Datetime(datetime_format='%m~%d~%Y'),
                                  'dates': Datetime},
                   time_index='ymd_special')

    assert dt.time_index == 'ymd_special'
    assert dt['dates'].logical_type == Datetime
    assert isinstance(dt['ymd_special'].logical_type, Datetime)
    assert isinstance(dt['mdy_special'].logical_type, Datetime)

    dt = dt.set_time_index('mdy_special')
    assert dt.time_index == 'mdy_special'

    df = pd.DataFrame({
        'mdy_special': pd.Series(['3&11&2000', '3&12&2000', '3&13&2000'], dtype='string'),
    })
    dt = DataTable(df)

    dt = dt.set_logical_types({'mdy_special': Datetime(datetime_format='%m&%d&%Y')})
    dt.time_index = 'mdy_special'
    assert isinstance(dt['mdy_special'].logical_type, Datetime)
    assert dt.time_index == 'mdy_special'


def test_natural_language_inference_with_config_options():
    dataframe = pd.DataFrame({
        'index': [0, 1, 2],
        'values': ["0123456", "01234567", "012345"]
    })

    ww.config.set_option('natural_language_threshold', 5)
    dt = DataTable(dataframe, name='dt_name')
    assert dt.columns['values'].logical_type == NaturalLanguage
    ww.config.reset_option('natural_language_threshold')


def test_to_pandas_copy(sample_df):
    dt = DataTable(sample_df)

    df_no_copy = dt.to_pandas()
    df_copy = dt.to_pandas(copy=True)

    assert df_no_copy is sample_df

    assert df_no_copy is not df_copy

    df_copy['test_col'] = pd.Series([1, 2, 3])
    assert 'test_col' in df_copy.columns
    assert 'test_col' not in df_no_copy.columns
    assert 'test_col' not in dt.columns


def test_describe_does_not_include_index():
    df = pd.DataFrame({'index_col': [0, 1, 2],
                       'values': [10, 20.3, 5]})
    dt = DataTable(df, index='index_col')
    stats_df = dt.describe()
    assert 'index_col' not in stats_df.columns


def test_data_table_describe_method():
    categorical_ltypes = [Categorical,
                          CountryCode,
                          Ordinal(order=('yellow', 'red', 'blue')),
                          SubRegionCode,
                          ZIPCode]
    boolean_ltypes = [Boolean]
    datetime_ltypes = [Datetime]
    formatted_datetime_ltypes = [Datetime(datetime_format='%Y~%m~%d')]
    timedelta_ltypes = [Timedelta]
    numeric_ltypes = [Double, Integer, WholeNumber]
    natural_language_ltypes = [EmailAddress, Filepath, FullName, IPAddress,
                               LatLong, PhoneNumber, URL]

    boolean_data = [True, False, True, True, False, True, np.nan, True]
    category_data = ['red', 'blue', 'red', np.nan, 'red', 'blue', 'red', 'yellow']
    datetime_data = pd.to_datetime(['2020-01-01',
                                    '2020-02-01',
                                    '2020-01-01 08:00',
                                    '2020-02-02 16:00',
                                    '2020-02-02 18:00',
                                    pd.NaT,
                                    '2020-02-01',
                                    '2020-01-02'])
    formatted_datetime_data = pd.Series(['2020~01~01',
                                         '2020~02~01',
                                         '2020~03~01',
                                         '2020~02~02',
                                         '2020~03~02',
                                         pd.NaT,
                                         '2020~02~01',
                                         '2020~01~02'])
    numeric_data = pd.Series([10, 20, 17, 32, np.nan, 1, 56, 10])
    natural_language_data = [
        'This is a natural language sentence',
        'Duplicate sentence.',
        'This line has numbers in it 000123.',
        'How about some symbols?!',
        'This entry contains two sentences. Second sentence.',
        'Duplicate sentence.',
        np.nan,
        'I am the last line',
    ]
    timedelta_data = datetime_data - pd.Timestamp('2020-01-01')

    expected_index = ['physical_type',
                      'logical_type',
                      'semantic_tags',
                      'count',
                      'nunique',
                      'nan_count',
                      'mean',
                      'mode',
                      'std',
                      'min',
                      'first_quartile',
                      'second_quartile',
                      'third_quartile',
                      'max',
                      'num_true',
                      'num_false']

    # Test categorical columns
    for ltype in categorical_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'category', 'custom_tag'},
            'count': 7,
            'nunique': 3,
            'nan_count': 1,
            'mode': 'red'}, name='col')
        df = pd.DataFrame({'col': category_data})
        dt = DataTable(df, logical_types={'col': ltype}, semantic_tags={'col': 'custom_tag'})
        stats_df = dt.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['col'].dropna())

    # Test boolean columns
    for ltype in boolean_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 7,
            'nan_count': 1,
            'mode': True,
            'num_true': 5,
            'num_false': 2}, name='col')
        expected_vals.name = 'col'
        df = pd.DataFrame({'col': boolean_data})
        dt = DataTable(df, logical_types={'col': ltype}, semantic_tags={'col': 'custom_tag'})
        stats_df = dt.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['col'].dropna())

    # Test datetime columns
    for ltype in datetime_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 7,
            'nunique': 6,
            'nan_count': 1,
            'mean': datetime_data.mean(),
            'mode': pd.to_datetime('2020-02-01'),
            'min': datetime_data.min(),
            'max': datetime_data.max()}, name='col')
        df = pd.DataFrame({'col': datetime_data})
        dt = DataTable(df, logical_types={'col': ltype}, semantic_tags={'col': 'custom_tag'})
        stats_df = dt.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['col'].dropna())

    # Test formatted datetime columns
    for ltype in formatted_datetime_ltypes:
        converted_to_datetime = pd.to_datetime(['2020-01-01',
                                                '2020-02-01',
                                                '2020-03-01',
                                                '2020-02-02',
                                                '2020-03-02',
                                                pd.NaT,
                                                '2020-02-01',
                                                '2020-01-02'])
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 7,
            'nunique': 6,
            'nan_count': 1,
            'mean': converted_to_datetime.mean(),
            'mode': pd.to_datetime('2020-02-01'),
            'min': converted_to_datetime.min(),
            'max': converted_to_datetime.max()}, name='formatted_col')
        df = pd.DataFrame({'formatted_col': formatted_datetime_data})
        dt = DataTable(df, logical_types={'formatted_col': ltype}, semantic_tags={'formatted_col': 'custom_tag'})
        stats_df = dt.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'formatted_col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['formatted_col'].dropna())

    # Test timedelta columns
    for ltype in timedelta_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 7,
            'nan_count': 1,
            'mode': pd.Timedelta('31days')}, name='col')
        df = pd.DataFrame({'col': timedelta_data})
        dt = DataTable(df, logical_types={'col': ltype}, semantic_tags={'col': 'custom_tag'})
        stats_df = dt.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['col'].dropna())

    # Test numeric columns
    for ltype in numeric_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'numeric', 'custom_tag'},
            'count': 7,
            'nunique': 6,
            'nan_count': 1,
            'mean': numeric_data.mean(),
            'mode': 10,
            'std': numeric_data.std(),
            'min': 1,
            'first_quartile': 10,
            'second_quartile': 17,
            'third_quartile': 26,
            'max': 56}, name='col')
        df = pd.DataFrame({'col': numeric_data})
        dt = DataTable(df, logical_types={'col': ltype}, semantic_tags={'col': 'custom_tag'})
        stats_df = dt.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['col'].dropna())

    # Test natural language columns
    for ltype in natural_language_ltypes:
        expected_vals = pd.Series({
            'physical_type': ltype.pandas_dtype,
            'logical_type': ltype,
            'semantic_tags': {'custom_tag'},
            'count': 7,
            'nan_count': 1,
            'mode': 'Duplicate sentence.'}, name='col')
        df = pd.DataFrame({'col': natural_language_data})
        dt = DataTable(df, logical_types={'col': ltype}, semantic_tags={'col': 'custom_tag'})
        stats_df = dt.describe()
        assert isinstance(stats_df, pd.DataFrame)
        assert set(stats_df.columns) == {'col'}
        assert stats_df.index.tolist() == expected_index
        pd.testing.assert_series_equal(expected_vals, stats_df['col'].dropna())


def test_datatable_describe_with_improper_tags():
    df = pd.DataFrame({'bool_col': [True, False, True, np.nan, True],
                       'text_col': ['one', 'two', 'three', 'four', 'five']})

    logical_types = {
        'bool_col': Boolean,
        'text_col': NaturalLanguage,
    }
    semantic_tags = {
        'bool_col': 'category',
        'text_col': 'numeric',
    }

    dt = DataTable(df, logical_types=logical_types, semantic_tags=semantic_tags)
    stats_df = dt.describe()

    # Make sure boolean stats were computed with improper 'category' tag
    assert stats_df['bool_col']['logical_type'] == Boolean
    assert stats_df['bool_col']['semantic_tags'] == {'category'}
    # Make sure numeric stats were not computed with improper 'numeric' tag
    assert stats_df['text_col']['semantic_tags'] == {'numeric'}
    assert stats_df['text_col'][['mean', 'std', 'min', 'max']].isnull().all()


def test_datatable_describe_with_no_semantic_tags():
    df = pd.DataFrame({'category_col': ['a', 'b', 'c', 'a', 'a'],
                       'num_col': [1, 3, 2, 4, 0]})

    logical_types = {
        'category_col': Categorical,
        'num_col': WholeNumber,
    }

    dt = DataTable(df, logical_types=logical_types, use_standard_tags=False)
    stats_df = dt.describe()
    assert dt['category_col'].semantic_tags == set()
    assert dt['num_col'].semantic_tags == set()

    # Make sure category stats were computed
    assert stats_df['category_col']['semantic_tags'] == set()
    assert stats_df['category_col']['nunique'] == 3
    # Make sure numeric stats were computed
    assert stats_df['num_col']['semantic_tags'] == set()
    assert stats_df['num_col']['mean'] == 2


def test_data_table_describe_with_include(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'age': ['numeric', 'age']
    }
    dt = DataTable(sample_df, semantic_tags=semantic_tags)

    col_name_df = dt.describe(include=['full_name'])
    assert col_name_df.shape == (16, 1)
    assert 'full_name', 'email' in col_name_df.columns

    semantic_tags_df = dt.describe(['tag1', 'tag2'])
    assert 'full_name' in col_name_df.columns
    assert len(semantic_tags_df.columns) == 2

    logical_types_df = dt.describe([Datetime, Boolean])
    assert 'signup_date', 'is_registered' in logical_types_df.columns
    assert len(logical_types_df.columns) == 2

    multi_params_df = dt.describe(['age', 'tag1', Datetime])
    expected = ['full_name', 'age', 'signup_date']
    for col_name in expected:
        assert col_name in multi_params_df.columns
    multi_params_df['full_name'].equals(col_name_df['full_name'])
    multi_params_df['full_name'].equals(dt.describe()['full_name'])


def test_data_table_describe_with_include_error(sample_df):
    dt = DataTable(sample_df)
    match = 'no columns matched the given include filters.'
    warning = 'The following selectors were not present in your DataTable: '

    with pytest.raises(ValueError, match=match):
        with pytest.warns(UserWarning, match=warning + 'wrongname'):
            dt.describe(include=['wrongname'])

    with pytest.warns(UserWarning, match=warning + 'tag4'):
        dt.describe(include=['email', 'tag4'])

    with pytest.raises(ValueError, match=match):
        with pytest.warns(UserWarning, match=warning + 'url'):
            dt.describe(include=[URL])


def test_data_table_handle_nans_for_mutual_info():
    df_nans = pd.DataFrame({
        'nans': pd.Series([None, None, None, None]),
        'ints': pd.Series([2, pd.NA, 5, 2], dtype='Int64'),
        'floats': pd.Series([3.3, None, 2.3, 1.3]),
        'bools': pd.Series([True, None, True, False]),
        'int_to_cat_nan': pd.Series([1, np.nan, 3, 1], dtype='category'),
        'str': pd.Series(['test', np.nan, 'test2', 'test']),
        'str_no_nan': pd.Series(['test', 'test2', 'test2', 'test']),
    })
    dt_nans = DataTable(df_nans)
    formatted_df = dt_nans._handle_nans_for_mutual_info(dt_nans.to_pandas(copy=True))

    assert isinstance(formatted_df, pd.DataFrame)

    assert 'nans' not in formatted_df.columns
    assert formatted_df['ints'].equals(pd.Series([2, 3, 5, 2], dtype='Int64'))
    assert formatted_df['floats'].equals(pd.Series([3.3, 2.3, 2.3, 1.3], dtype='float'))
    assert formatted_df['bools'].equals(pd.Series([True, True, True, False], dtype='category'))
    assert formatted_df['int_to_cat_nan'].equals(pd.Series([1, 1, 3, 1], dtype='category'))
    assert formatted_df['str'].equals(pd.Series(['test', 'test', 'test2', 'test'], dtype='category'))
    assert formatted_df['str_no_nan'].equals(pd.Series(['test', 'test2', 'test2', 'test'], dtype='category'))


def test_data_table_make_categorical_for_mutual_info():
    df = pd.DataFrame({
        'ints1': pd.Series([1, 2, 3, 2]),
        'ints2': pd.Series([1, 100, 1, 100]),
        'bools': pd.Series([True, False, True, False]),
        'categories': pd.Series(['test', 'test2', 'test2', 'test'])
    })
    dt = DataTable(df)
    formatted_num_bins_df = dt._make_categorical_for_mutual_info(dt.to_pandas(copy=True), num_bins=4)

    assert isinstance(formatted_num_bins_df, pd.DataFrame)

    assert formatted_num_bins_df['ints1'].equals(pd.Series([0, 1, 3, 1], dtype='int8'))
    assert formatted_num_bins_df['ints2'].equals(pd.Series([0, 1, 0, 1], dtype='int8'))
    assert formatted_num_bins_df['bools'].equals(pd.Series([1, 0, 1, 0], dtype='int8'))
    assert formatted_num_bins_df['categories'].equals(pd.Series([0, 1, 1, 0], dtype='int8'))


def test_data_table_get_mutual_information():
    df_same_mi = pd.DataFrame({
        'ints': pd.Series([2, pd.NA, 5, 2], dtype='Int64'),
        'floats': pd.Series([1, None, 100, 1]),
        'nans': pd.Series([None, None, None, None]),
        'nat_lang': pd.Series(['this is a very long sentence inferred as a string', None, 'test', 'test']),
        'date': pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'])
    })
    dt_same_mi = DataTable(df_same_mi, logical_types={'date': Datetime(datetime_format='%Y-%m-%d')})

    mi = dt_same_mi.get_mutual_information()

    cols_used = set(np.unique(mi[['column_1', 'column_2']].values))
    assert 'nans' not in cols_used
    assert 'nat_lang' not in cols_used
    assert 'date' not in cols_used
    assert mi.shape[0] == 1
    assert mi_between_cols('floats', 'ints', mi) == 1.0

    df = pd.DataFrame({
        'ints': pd.Series([1, 2, 3]),
        'bools': pd.Series([True, False, True]),
        'strs': pd.Series(['hi', 'hi', 'hi'])
    })
    dt = DataTable(df)
    original_df = dt.to_pandas(copy=True)

    mi = dt.get_mutual_information()
    assert mi.shape[0] == 3
    np.testing.assert_almost_equal(mi_between_cols('ints', 'bools', mi), 0.734, 3)
    np.testing.assert_almost_equal(mi_between_cols('ints', 'strs', mi), 0.0, 3)
    np.testing.assert_almost_equal(mi_between_cols('strs', 'bools', mi), 0, 3)

    mi_many_rows = dt.get_mutual_information(nrows=100000)
    pd.testing.assert_frame_equal(mi, mi_many_rows)

    mi = dt.get_mutual_information(nrows=1)
    assert mi.shape[0] == 3
    assert (mi['mutual_info'] == 1.0).all()

    mi = dt.get_mutual_information(num_bins=2)
    assert mi.shape[0] == 3
    np.testing.assert_almost_equal(mi_between_cols('bools', 'ints', mi), .274, 3)
    np.testing.assert_almost_equal(mi_between_cols('strs', 'ints', mi), 0, 3)
    np.testing.assert_almost_equal(mi_between_cols('bools', 'strs', mi), 0, 3)

    # Confirm that none of this changed the DataTable's underlying df
    pd.testing.assert_frame_equal(dt.to_pandas(), original_df)


def test_mutual_info_does_not_include_index():
    df = pd.DataFrame({'index_col': pd.Series([0, 1, 2], dtype='string'),
                       'values': [10, 20.3, 5]})
    dt = DataTable(df, index='index_col')
    mi = dt.get_mutual_information()

    assert mi.shape[0] == 0


def test_mutual_info_sort():
    df = pd.DataFrame({
        'ints': pd.Series([1, 2, 3]),
        'bools': pd.Series([True, False, True]),
        'strs2': pd.Series(['bye', 'hi', 'bye']),

        'strs': pd.Series(['hi', 'hi', 'hi'])

    })
    dt = DataTable(df)
    mi = dt.get_mutual_information()

    for i in range(len(mi['mutual_info']) - 1):
        assert mi['mutual_info'].iloc[i] >= mi['mutual_info'].iloc[i + 1]


def test_make_index(sample_df):
    dt = DataTable(sample_df, index='new_index', make_index=True)
    assert dt.index == 'new_index'
    assert 'new_index' in dt._dataframe.columns
    assert dt._dataframe['new_index'].unique
    assert dt._dataframe['new_index'].is_monotonic
    assert 'index' in dt.columns['new_index'].semantic_tags


def test_numeric_time_index_dtypes():
    df = pd.DataFrame({
        'whole_numbers': pd.Series([1, 2, 3], dtype='int8'),
        'floats': pd.Series([1, 2, 3], dtype='float'),
        'ints': pd.Series([1, -2, 3], dtype='Int64'),
        'with_null': pd.Series([1, 2, pd.NA], dtype='Int64'),
    })

    dt = DataTable(df, time_index='whole_numbers')
    date_col = dt['whole_numbers']
    assert dt.time_index == 'whole_numbers'
    assert date_col.logical_type == WholeNumber
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    dt = dt.set_time_index('floats')
    date_col = dt['floats']
    assert dt.time_index == 'floats'
    assert date_col.logical_type == Double
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    dt = dt.set_time_index('ints')
    date_col = dt['ints']
    assert dt.time_index == 'ints'
    assert date_col.logical_type == Integer
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    dt = dt.set_time_index('with_null')
    date_col = dt['with_null']
    assert dt.time_index == 'with_null'
    assert date_col.logical_type == WholeNumber
    assert date_col.semantic_tags == {'time_index', 'numeric'}


def test_numeric_index_strings():
    df = pd.DataFrame({'strs': pd.Series(['1', '2', '3']),
                       'ints': pd.Series([1, 2, 3])})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        DataTable(df, time_index='strs')

    error_msg = 'Error converting datatype for column strs from type object to type Int64. Please confirm the underlying data is consistent with logical type Integer.'
    with pytest.raises(TypeError, match=error_msg):
        DataTable(df, time_index='strs', logical_types={'strs': 'Integer'})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        DataTable(df, time_index='ints', logical_types={'ints': 'Categorical'})

    dt = DataTable(df, time_index='strs', logical_types={'strs': 'Double'})
    date_col = dt['strs']
    assert dt.time_index == 'strs'
    assert date_col.logical_type == Double
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    dt = DataTable(df, logical_types={'strs': 'Double'})
    dt = dt.set_time_index('strs')
    date_col = dt['strs']
    assert dt.time_index == 'strs'
    assert date_col.logical_type == Double
    assert date_col.semantic_tags == {'time_index', 'numeric'}
