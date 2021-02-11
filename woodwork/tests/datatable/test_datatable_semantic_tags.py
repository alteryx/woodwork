import pytest

from woodwork import DataTable
from woodwork.logical_types import (
    Datetime,
    EmailAddress,
    FullName,
    Integer,
    PhoneNumber,
    ZIPCode
)


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
    new_dt = dt.set_types(semantic_tags=new_tags)
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
    dt = dt.set_types(semantic_tags=new_tags)
    assert dt.columns['id'].semantic_tags == {'index', 'new_tag'}
    dt = dt.set_types(semantic_tags=new_tags, retain_index_tags=False)
    assert dt.columns['id'].semantic_tags == {'new_tag'}


def test_set_semantic_tags_with_time_index(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', use_standard_tags=False)
    assert dt.columns['signup_date'].semantic_tags == {'time_index'}

    new_tags = {
        'signup_date': 'new_tag',
    }
    dt = dt.set_types(semantic_tags=new_tags)
    assert dt.columns['signup_date'].semantic_tags == {'time_index', 'new_tag'}
    dt = dt.set_types(semantic_tags=new_tags, retain_index_tags=False)
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


def test_select_semantic_tags(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', name='dt_name')
    dt = dt.set_types(semantic_tags={
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


def test_select_semantic_tags_no_match(sample_df):
    dt = DataTable(sample_df, time_index='signup_date', index='id', name='dt_name')
    dt = dt.set_types(logical_types={
        'full_name': FullName,
        'email': EmailAddress,
        'phone_number': PhoneNumber,
        'signup_date': Datetime(datetime_format='%Y-%m-%d'),
    })
    dt = dt.set_types(semantic_tags={
        'full_name': ['new_tag', 'tag2'],
        'age': 'numeric',
        'signup_date': 'date_of_birth',
        'email': 'tag2'
    })

    assert len(dt.select(['doesnt_exist']).columns) == 0

    dt_multiple_unused = dt.select(['doesnt_exist', 'boolean', 'category', PhoneNumber])
    assert len(dt_multiple_unused.columns) == 2

    dt_unused_ltype = dt.select(['date_of_birth', 'doesnt_exist', ZIPCode, Integer])
    assert len(dt_unused_ltype.columns) == 3


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
