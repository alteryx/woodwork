import pandas as pd

import woodwork as ww
from woodwork.schema import Schema
from woodwork.logical_types import (
    Boolean,
    Datetime,
    Integer,
    LogicalType,
    NaturalLanguage
)


def test_schema_physical_types(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types)
    assert isinstance(schema.physical_types, dict)
    assert set(schema.physical_types.keys()) == set(sample_column_names)
    for k, v in schema.physical_types.items():
        assert isinstance(k, str)
        assert v == schema.columns[k]['logical_type'].pandas_dtype


def test_schema_logical_types(sample_column_names, sample_inferred_logical_types):
    schema = Schema(sample_column_names, sample_inferred_logical_types)
    assert isinstance(schema.logical_types, dict)
    assert set(schema.logical_types.keys()) == set(sample_column_names)
    for k, v in schema.logical_types.items():
        assert isinstance(k, str)
        assert k in sample_column_names
        assert v in ww.type_system.registered_types
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
        assert isinstance(k, str)
        assert k in sample_column_names
        assert isinstance(v, set)
        assert v == schema.columns[k]['semantic_tags']


def test_schema_types(sample_column_names, sample_inferred_logical_types):
    sample_column_names.append('formatted_date')

    ymd_format = Datetime(datetime_format='%Y~%m~%d')
    schema = Schema(sample_column_names, logical_types={**sample_inferred_logical_types, **{'formatted_date': ymd_format}})

    returned_types = schema.types
    assert isinstance(returned_types, pd.DataFrame)
    assert 'Physical Type' in returned_types.columns
    assert 'Logical Type' in returned_types.columns
    assert 'Semantic Tag(s)' in returned_types.columns
    assert returned_types.shape[1] == 3
    assert len(returned_types.index) == len(sample_column_names)
    assert all([col_dict['logical_type'] in ww.type_system.registered_types or isinstance(col_dict['logical_type'], LogicalType) for col_dict in schema.columns.values()])
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
    for tag in returned_types['Semantic Tag(s)']:
        assert isinstance(tag, str)


def test_schema_repr(small_df):
    schema = Schema(list(small_df.columns), logical_types={'sample_datetime_series': 'Datetime'})

    schema_repr = repr(schema)
    expected_repr = '                         Physical Type Logical Type Semantic Tag(s)\nColumn                                                             \nsample_datetime_series  datetime64[ns]     Datetime              []'
    assert schema_repr == expected_repr

    schema_html_repr = schema._repr_html_()
    expected_repr = '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Physical Type</th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>sample_datetime_series</th>\n      <td>datetime64[ns]</td>\n      <td>Datetime</td>\n      <td>[]</td>\n    </tr>\n  </tbody>\n</table>'
    assert schema_html_repr == expected_repr


def test_schema_repr_empty():
    schema = Schema([], {})
    assert repr(schema) == 'Empty Schema'

    assert schema._repr_html_() == 'Empty Schema'


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
                                logical_types={**sample_inferred_logical_types, **{'full_name': 'categorical'}},
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

    schema.metadata = metadata
    assert schema.metadata == metadata

    schema = Schema(sample_column_names, sample_inferred_logical_types,
                    table_metadata=metadata, time_index='signup_date')
    assert schema.metadata == metadata

    new_data = {'date_created': '1/1/19', 'created_by': 'user1'}
    schema.metadata = {**metadata, **new_data}
    assert schema.metadata == {'secondary_time_index': {'is_registered': 'age'},
                               'date_created': '1/1/19',
                               'created_by': 'user1'}

    schema.metadata.pop('created_by')
    assert schema.metadata == {'secondary_time_index': {'is_registered': 'age'}, 'date_created': '1/1/19'}

    schema.metadata['number'] = 1012034
    assert schema.metadata == {'number': 1012034,
                               'secondary_time_index': {'is_registered': 'age'},
                               'date_created': '1/1/19'}


def test_schema_column_metadata(sample_column_names, sample_inferred_logical_types):
    column_metadata = {'metadata_field': [1, 2, 3], 'created_by': 'user0'}

    schema = Schema(sample_column_names, sample_inferred_logical_types)
    assert schema.columns['id']['metadata'] == {}

    schema = Schema(sample_column_names, sample_inferred_logical_types, column_metadata={'id': column_metadata})
    assert schema.columns['id']['metadata'] == column_metadata

    new_metadata = {'date_created': '1/1/19', 'created_by': 'user1'}

    schema.columns['id']['metadata'] = {**schema.columns['id']['metadata'], **new_metadata}
    assert schema.columns['id']['metadata'] == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3], 'created_by': 'user1'}

    schema.columns['id']['metadata'].pop('created_by')
    assert schema.columns['id']['metadata'] == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3]}

    schema.columns['id']['metadata']['number'] = 1012034
    assert schema.columns['id']['metadata'] == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3], 'number': 1012034}
