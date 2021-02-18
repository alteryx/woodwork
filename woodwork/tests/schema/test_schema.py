import inspect
import re

import pandas as pd
import pytest

from woodwork import type_system
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    FullName,
    Integer,
    NaturalLanguage
)
from woodwork.schema import Schema


def test_schema_physical_types(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types)
    assert isinstance(schema.physical_types, dict)
    assert set(schema.physical_types.keys()) == set(sample_column_names)
    for k, v in schema.physical_types.items():
        assert v == schema.columns[k]['logical_type'].pandas_dtype


def test_schema_logical_types(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types)
    assert isinstance(schema.logical_types, dict)
    assert set(schema.logical_types.keys()) == set(sample_column_names)
    for k, v in schema.logical_types.items():
        assert v == schema.columns[k]['logical_type']


def test_schema_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'age': ['numeric', 'age']
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types, semantic_tags=semantic_tags)
    assert isinstance(schema.semantic_tags, dict)
    assert set(schema.semantic_tags.keys()) == set(sample_column_names)
    for k, v in schema.semantic_tags.items():
        assert isinstance(v, set)
        assert v == schema.columns[k]['semantic_tags']


def test_schema_types(sample_column_names, sample_inferred_logical_types):
    sample_column_names.append('formatted_date')

    ymd_format = Datetime(datetime_format='%Y~%m~%d')
    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, 'formatted_date': ymd_format})

    returned_types = schema.types
    assert isinstance(returned_types, pd.DataFrame)
    assert 'Physical Type' in returned_types.columns
    assert 'Logical Type' in returned_types.columns
    assert 'Semantic Tag(s)' in returned_types.columns
    assert returned_types.shape[1] == 3
    assert len(returned_types.index) == len(sample_column_names)
    correct_logical_types = {
        'id': Integer,
        'full_name': NaturalLanguage,
        'email': NaturalLanguage,
        'phone_number': NaturalLanguage,
        'age': Integer,
        'signup_date': Datetime,
        'is_registered': Boolean,
        'formatted_date': ymd_format
    }
    correct_logical_types = pd.Series(list(correct_logical_types.values()),
                                      index=list(correct_logical_types.keys()))
    assert correct_logical_types.equals(returned_types['Logical Type'])

    correct_semantic_tags = {
        'id': "['numeric']",
        'full_name': "[]",
        'email': "[]",
        'phone_number': "[]",
        'age': "['numeric']",
        'signup_date': "[]",
        'is_registered': "[]",
        'formatted_date': "[]",
    }
    correct_semantic_tags = pd.Series(list(correct_semantic_tags.values()),
                                      index=list(correct_semantic_tags.keys()))
    assert correct_semantic_tags.equals(returned_types['Semantic Tag(s)'])


def test_schema_repr(small_df):
    schema = Schema(list(small_df.columns), logical_types={'sample_datetime_series': Datetime})

    schema_repr = repr(schema)
    expected_repr = '                         Physical Type Logical Type Semantic Tag(s)\nColumn                                                             \nsample_datetime_series  datetime64[ns]     Datetime              []'
    assert schema_repr == expected_repr

    schema_html_repr = schema._repr_html_()
    expected_repr = '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Physical Type</th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>sample_datetime_series</th>\n      <td>datetime64[ns]</td>\n      <td>Datetime</td>\n      <td>[]</td>\n    </tr>\n  </tbody>\n</table>'
    assert schema_html_repr == expected_repr


def test_schema_repr_empty():
    schema = Schema([], {})
    assert repr(schema) == 'Empty DataFrame\nColumns: [Physical Type, Logical Type, Semantic Tag(s)]\nIndex: []'

    assert schema._repr_html_() == '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Physical Type</th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n  </tbody>\n</table>'


def test_schema_equality(sample_column_names, sample_inferred_logical_types):
    schema_basic = Schema(sample_column_names, sample_inferred_logical_types)
    schema_basic2 = Schema(sample_column_names, sample_inferred_logical_types)
    schema_names = Schema(sample_column_names, sample_inferred_logical_types, name='test')

    assert schema_basic != schema_names
    assert schema_basic == schema_basic2

    missing_col_names = sample_column_names[1:]
    missing_logical_types = sample_inferred_logical_types.copy()
    missing_logical_types.pop('id')

    schema_missing_col = Schema(missing_col_names, missing_logical_types)
    assert schema_basic != schema_missing_col

    schema_index = Schema(sample_column_names, sample_inferred_logical_types, index='id')
    schema_time_index = Schema(sample_column_names, sample_inferred_logical_types, time_index='signup_date')

    assert schema_basic != schema_index
    assert schema_index != schema_time_index

    schema_numeric_time_index = Schema(sample_column_names, sample_inferred_logical_types, time_index='id')

    assert schema_time_index != schema_numeric_time_index

    schema_with_ltypes = Schema(sample_column_names,
                                logical_types={**sample_inferred_logical_types, 'full_name': Categorical},
                                time_index='signup_date')
    assert schema_with_ltypes != schema_time_index

    schema_with_metadata = Schema(sample_column_names, sample_inferred_logical_types, index='id', table_metadata={'created_by': 'user1'})
    assert Schema(sample_column_names, sample_inferred_logical_types, index='id') != schema_with_metadata
    assert Schema(sample_column_names,
                  sample_inferred_logical_types,
                  index='id',
                  table_metadata={'created_by': 'user1'}) == schema_with_metadata
    assert Schema(sample_column_names,
                  sample_inferred_logical_types,
                  index='id',
                  table_metadata={'created_by': 'user2'}) != schema_with_metadata


def test_schema_table_metadata(sample_column_names, sample_inferred_logical_types):
    metadata = {'secondary_time_index': {'is_registered': 'age'}, 'date_created': '11/13/20'}

    schema = Schema(sample_column_names, sample_inferred_logical_types)
    assert schema.metadata == {}

    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    table_metadata=metadata, time_index='signup_date')
    assert schema.metadata == metadata


def test_schema_column_metadata(sample_column_names, sample_inferred_logical_types):
    column_metadata = {'metadata_field': [1, 2, 3], 'created_by': 'user0'}

    schema = Schema(sample_column_names, sample_inferred_logical_types)
    assert schema.columns['id']['metadata'] == {}

    schema = Schema(sample_column_names, sample_inferred_logical_types, column_metadata={'id': column_metadata})
    assert schema.columns['id']['metadata'] == column_metadata


def test_filter_schema_cols(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    time_index='signup_date',
                    index='id',
                    name='dt_name')

    filtered = schema._filter_cols(include=Datetime)
    assert filtered == ['signup_date']

    filtered = schema._filter_cols(include='email', col_names=True)
    assert filtered == ['email']

    filtered_log_type_string = schema._filter_cols(include='NaturalLanguage')
    filtered_log_type = schema._filter_cols(include=NaturalLanguage)
    expected = {'full_name', 'email', 'phone_number'}
    assert filtered_log_type == filtered_log_type_string
    assert set(filtered_log_type) == expected

    filtered_semantic_tag = schema._filter_cols(include='numeric')
    assert filtered_semantic_tag == ['age']

    filtered_multiple_overlap = schema._filter_cols(include=['NaturalLanguage', 'email'], col_names=True)
    expected = ['full_name', 'phone_number', 'email']
    for col in filtered_multiple_overlap:
        assert col in expected


def test_filter_schema_cols_no_matches(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    time_index='signup_date',
                    index='id',
                    name='dt_name')

    filter_no_matches = schema._filter_cols(include='nothing')
    assert filter_no_matches == []

    filter_empty_list = schema._filter_cols(include=[])
    assert filter_empty_list == []

    filter_non_string = schema._filter_cols(include=1)
    assert filter_non_string == []


def test_filter_schema_errors(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    time_index='signup_date',
                    index='id',
                    name='dt_name')

    err_msg = "Invalid selector used in include: {} must be a string, uninstantiated and registered LogicalType, or valid column name"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(include=['boolean', 'index', Double, {}])

    err_msg = "Invalid selector used in include: {} must be a string, uninstantiated and registered LogicalType, or valid column name"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(include=['boolean', 'index', Double, {}], col_names=True)

    err_msg = "Invalid selector used in include: Datetime cannot be instantiated"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(Datetime())

    type_system.remove_type(EmailAddress)
    err_msg = "Specified LogicalType selector EmailAddress is not registered in Woodwork's type system."
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(EmailAddress)

    err_msg = "Invalid selector used in include: EmailAddress must be a string, uninstantiated and registered LogicalType, or valid column name"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(EmailAddress())
    type_system.reset_defaults()


def test_filter_schema_overlap_name_and_type(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types)

    filter_name_ltype_overlap = schema._filter_cols(include='full_name')
    assert filter_name_ltype_overlap == []

    filter_overlap_with_name = schema._filter_cols(include='full_name', col_names=True)
    assert filter_overlap_with_name == ['full_name']

    schema = Schema(sample_column_names,
                    {**sample_inferred_logical_types, 'full_name': Categorical, 'age': FullName},
                    semantic_tags={'id': 'full_name'})

    filter_tag_and_ltype = schema._filter_cols(include='full_name')
    assert set(filter_tag_and_ltype) == {'id', 'age'}

    filter_all_three = schema._filter_cols(include='full_name', col_names=True)
    assert set(filter_all_three) == {'id', 'age', 'full_name'}


def test_filter_schema_non_string_cols():
    schema = Schema(column_names=[0, 1, 2, 3], logical_types={0: Integer, 1: Categorical, 2: NaturalLanguage, 3: Double})

    filter_types_and_tags = schema._filter_cols(include=[Integer, 'category'])
    assert filter_types_and_tags == [0, 1]

    filter_by_name = schema._filter_cols(include=[0, 1], col_names=True)
    assert filter_by_name == [0, 1]


def test_get_subset_schema(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types)
    new_schema = schema._get_subset_schema(sample_column_names[1:4])
    for col in new_schema.columns:
        assert new_schema.semantic_tags[col] == schema.semantic_tags[col]
        assert new_schema.logical_types[col] == schema.logical_types[col]


def test_get_subset_schema_all_params(sample_column_names, sample_inferred_logical_types):
    # The first element is self, so it won't be included in kwargs
    possible_schema_params = inspect.getfullargspec(Schema.__init__)[0][1:]

    kwargs = {
        'column_names': sample_column_names,
        'logical_types': {**sample_inferred_logical_types, 'email': EmailAddress},
        'name': 'test_dt',
        'index': 'id',
        'time_index': 'signup_date',
        'semantic_tags': {'age': 'test_tag'},
        'table_metadata': {'created_by': 'user1'},
        'column_metadata': {'phone_number': {'format': 'xxx-xxx-xxxx'}},
        'use_standard_tags': False,
        'column_descriptions': {'age': 'this is a description'}
    }

    # Confirm all possible params to Schema init are present with non-default values where possible
    assert set(possible_schema_params) == set(kwargs.keys())

    schema = Schema(**kwargs)
    copy_schema = schema._get_subset_schema(sample_column_names)

    assert schema == copy_schema
    assert schema is not copy_schema


def test_set_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        'full_name': 'tag1',
        'age': ['numeric', 'age']
    }
    expected_tags = {
        'full_name': {'tag1'},
        'age': {'numeric', 'age'}
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types, semantic_tags=semantic_tags)
    assert schema.semantic_tags['full_name'] == expected_tags['full_name']
    assert schema.semantic_tags['age'] == expected_tags['age']

    new_tags = {
        'full_name': ['new_tag'],
        'age': 'numeric',
    }
    schema.set_types(semantic_tags=new_tags)

    assert schema.semantic_tags['full_name'] == {'new_tag'}
    assert schema.semantic_tags['age'] == {'numeric'}


def test_set_semantic_tags_with_index(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='id', use_standard_tags=False)
    assert schema.semantic_tags['id'] == {'index'}

    new_tags = {
        'id': 'new_tag',
    }
    schema.set_types(semantic_tags=new_tags)
    assert schema.semantic_tags['id'] == {'index', 'new_tag'}

    schema.set_types(semantic_tags=new_tags, retain_index_tags=False)
    assert schema.semantic_tags['id'] == {'new_tag'}


def test_set_semantic_tags_with_time_index(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    time_index='signup_date', use_standard_tags=False)
    assert schema.semantic_tags['signup_date'] == {'time_index'}

    new_tags = {
        'signup_date': 'new_tag',
    }
    schema.set_types(semantic_tags=new_tags)
    assert schema.semantic_tags['signup_date'] == {'time_index', 'new_tag'}

    schema.set_types(semantic_tags=new_tags, retain_index_tags=False)
    assert schema.semantic_tags['signup_date'] == {'new_tag'}


def test_add_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        'full_name': 'tag1',
        'age': ['numeric', 'age']
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    semantic_tags=semantic_tags, use_standard_tags=False,
                    index='id')

    new_tags = {
        'full_name': ['list_tag'],
        'age': 'str_tag',
        'id': {'set_tag'}
    }
    schema.add_semantic_tags(new_tags)

    assert schema.semantic_tags['full_name'] == {'tag1', 'list_tag'}
    assert schema.semantic_tags['age'] == {'numeric', 'age', 'str_tag'}
    assert schema.semantic_tags['id'] == {'set_tag', 'index'}


def test_reset_all_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        'full_name': 'tag1',
        'age': 'age'
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types, semantic_tags=semantic_tags, use_standard_tags=True)

    schema.reset_semantic_tags()
    assert schema.semantic_tags['full_name'] == set()
    assert schema.semantic_tags['age'] == {'numeric'}


def test_reset_semantic_tags_with_index(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        'id': 'tag1',
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='id',
                    semantic_tags=semantic_tags,
                    use_standard_tags=False)
    assert schema.semantic_tags['id'] == {'index', 'tag1'}

    schema.reset_semantic_tags('id', retain_index_tags=True)
    assert schema.semantic_tags['id'] == {'index'}

    schema.reset_semantic_tags('id')
    assert schema.semantic_tags['id'] == set()


def test_reset_semantic_tags_with_time_index(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        'signup_date': 'tag1',
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    time_index='signup_date',
                    semantic_tags=semantic_tags,
                    use_standard_tags=False)
    assert schema.semantic_tags['signup_date'] == {'time_index', 'tag1'}

    schema.reset_semantic_tags('signup_date', retain_index_tags=True)
    assert schema.semantic_tags['signup_date'] == {'time_index'}

    schema.reset_semantic_tags('signup_date')
    assert schema.semantic_tags['signup_date'] == set()


def test_reset_semantic_tags_invalid_column(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types,)
    error_msg = "Input contains columns that are not present in dataframe: 'invalid_column'"
    with pytest.raises(LookupError, match=error_msg):
        schema.reset_semantic_tags('invalid_column')


def test_remove_semantic_tags(sample_column_names, sample_inferred_logical_types):
    semantic_tags = {
        'full_name': ['tag1', 'tag2', 'tag3'],
        'age': ['numeric', 'age'],
        'id': ['tag1', 'tag2']
    }
    schema = Schema(sample_column_names, sample_inferred_logical_types, semantic_tags=semantic_tags, use_standard_tags=False)
    tags_to_remove = {
        'full_name': ['tag1', 'tag3'],
        'age': 'numeric',
        'id': {'tag1'}
    }
    schema.remove_semantic_tags(tags_to_remove)
    assert schema.semantic_tags['full_name'] == {'tag2'}
    assert schema.semantic_tags['age'] == {'age'}
    assert schema.semantic_tags['id'] == {'tag2'}


def test_raises_error_setting_index_tag_directly(sample_column_names, sample_inferred_logical_types):
    error_msg = re.escape("Cannot add 'index' tag directly for column id. To set a column as the index, "
                          "use DataFrame.ww.set_index() instead.")

    schema = Schema(sample_column_names, sample_inferred_logical_types)

    with pytest.raises(ValueError, match=error_msg):
        schema.add_semantic_tags({'id': 'index'})
    with pytest.raises(ValueError, match=error_msg):
        schema.set_types(semantic_tags={'id': 'index'})


def test_raises_error_setting_time_index_tag_directly(sample_column_names, sample_inferred_logical_types):
    error_msg = re.escape("Cannot add 'time_index' tag directly for column signup_date. To set a column as the time index, "
                          "use DataFrame.ww.set_time_index() instead.")
    schema = Schema(sample_column_names, sample_inferred_logical_types)

    with pytest.raises(ValueError, match=error_msg):
        schema.add_semantic_tags({'signup_date': 'time_index'})
    with pytest.raises(ValueError, match=error_msg):
        schema.set_types(semantic_tags={'signup_date': 'time_index'})


def test_removes_index_via_tags(sample_column_names, sample_inferred_logical_types):
    # Check setting tags
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='id', use_standard_tags=True)
    schema.set_types(semantic_tags={'id': 'new_tag'}, retain_index_tags=False)
    assert schema.semantic_tags['id'] == {'numeric', 'new_tag'}
    assert schema.index is None

    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='id', use_standard_tags=False)
    schema.set_types(semantic_tags={'id': 'new_tag'}, retain_index_tags=False)
    assert schema.semantic_tags['id'] == {'new_tag'}
    assert schema.index is None

    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='full_name', use_standard_tags=True)
    schema.set_types(semantic_tags={'full_name': 'new_tag'}, retain_index_tags=False)
    assert schema.semantic_tags['full_name'] == {'new_tag'}
    assert schema.index is None

    # Check removing tags
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='id', use_standard_tags=True)
    schema.remove_semantic_tags(semantic_tags={'id': 'index'})
    assert schema.semantic_tags['id'] == {'numeric'}
    assert schema.index is None

    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='id', use_standard_tags=False)
    schema.remove_semantic_tags(semantic_tags={'id': 'index'})
    assert schema.semantic_tags['id'] == set()
    assert schema.index is None

    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='full_name', use_standard_tags=True)
    schema.remove_semantic_tags(semantic_tags={'full_name': 'index'})
    assert schema.semantic_tags['full_name'] == set()
    assert schema.index is None

    # Check resetting tags
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='id', use_standard_tags=True)
    schema.reset_semantic_tags('id')
    assert schema.semantic_tags['id'] == {'numeric'}
    assert schema.index is None

    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='id', use_standard_tags=False)
    schema.reset_semantic_tags('id')
    assert schema.semantic_tags['id'] == set()
    assert schema.index is None

    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    index='full_name', use_standard_tags=True)
    schema.reset_semantic_tags('full_name')
    assert schema.semantic_tags['full_name'] == set()
    assert schema.index is None


def test_removes_time_index_via_tags(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types, time_index='signup_date')
    schema.set_types(semantic_tags={'signup_date': 'new_tag'}, retain_index_tags=False)
    assert schema.semantic_tags['signup_date'] == {'new_tag'}
    assert schema.time_index is None

    schema = Schema(sample_column_names, sample_inferred_logical_types, time_index='signup_date')
    schema.remove_semantic_tags(semantic_tags={'signup_date': 'time_index'})
    assert schema.semantic_tags['signup_date'] == set()
    assert schema.time_index is None

    schema = Schema(sample_column_names, sample_inferred_logical_types, time_index='signup_date')
    schema.reset_semantic_tags('signup_date')
    assert schema.semantic_tags['signup_date'] == set()
    assert schema.time_index is None
