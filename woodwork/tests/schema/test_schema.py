import inspect

import pandas as pd
import pytest

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


def test_filter_schema_errors(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    time_index='signup_date',
                    index='id',
                    name='dt_name')

    err_msg = "Invalid selector used in include: 1 must be a string, LogicalType, or valid column name"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(include=['boolean', 'index', Double, 1])

    err_msg = "Invalid selector used in include: 1 must be a string, LogicalType, or valid column name"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(include=['boolean', 'index', Double, 1], col_names=True)

    err_msg = "Invalid selector used in include: Datetime cannot be instantiated"
    with pytest.raises(TypeError, match=err_msg):
        schema._filter_cols(Datetime())


def test_filter_schema_overlap_name_and_type(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names,
                    {**sample_inferred_logical_types, 'full_name': Categorical, 'age': FullName},
                    semantic_tags={'id': 'is_registered'})

    filter_full_name_ltype = schema._filter_cols(include='full_name')
    assert filter_full_name_ltype == ['age']

    filter_full_name_tag = schema._filter_cols(include='is_registered')
    assert filter_full_name_tag == ['id']

    filter_full_name_with_name_cols = schema._filter_cols(include=['full_name', 'is_registered'], col_names=True)
    # Since logical type and semantic tag are given priority, we don't get the columns with those names
    assert set(filter_full_name_with_name_cols) == {'age', 'id'}


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
    possible_dt_params = inspect.getfullargspec(Schema.__init__)[0][1:]

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
    assert set(possible_dt_params) == set(kwargs.keys())

    schema = Schema(**kwargs)
    copy_schema = schema._get_subset_schema(sample_column_names)

    assert schema == copy_schema
