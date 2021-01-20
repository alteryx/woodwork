import pandas as pd

import woodwork as ww
from woodwork import Schema
from woodwork.logical_types import (
    Boolean,
    Datetime,
    Integer,
    LogicalType,
    NaturalLanguage
)
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
dask_delayed = import_or_none('dask.delayed')
ks = import_or_none('databricks.koalas')


def test_schema_physical_types(sample_df):
    # --> move the next three to schema (not init) tests
    dt = Schema(sample_df)
    assert isinstance(dt.physical_types, dict)
    assert set(dt.physical_types.keys()) == set(sample_df.columns)
    for k, v in dt.physical_types.items():
        assert isinstance(k, str)
        assert v == sample_df[k].dtype


def test_schema_logical_types(sample_df):
    dt = Schema(sample_df)
    assert isinstance(dt.logical_types, dict)
    assert set(dt.logical_types.keys()) == set(sample_df.columns)
    for k, v in dt.logical_types.items():
        assert isinstance(k, str)
        assert k in sample_df.columns
        assert v in ww.type_system.registered_types
        assert v == dt.columns[k]['logical_type']


def test_schema_semantic_tags(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'age': ['numeric', 'age']
    }
    dt = Schema(sample_df, semantic_tags=semantic_tags)
    assert isinstance(dt.semantic_tags, dict)
    assert set(dt.semantic_tags.keys()) == set(sample_df.columns)
    for k, v in dt.semantic_tags.items():
        assert isinstance(k, str)
        assert k in sample_df.columns
        assert isinstance(v, set)
        assert v == dt.columns[k]['semantic_tags']


def test_schema_types(sample_df):
    new_dates = ["2019~01~01", "2019~01~02", "2019~01~03", "2019~01~04"]
    if dd and isinstance(sample_df, dd.DataFrame):
        sample_df['formatted_date'] = pd.Series(new_dates)
    else:
        sample_df['formatted_date'] = new_dates
    ymd_format = Datetime(datetime_format='%Y~%m~%d')
    dt = Schema(sample_df, logical_types={'formatted_date': ymd_format})
    returned_types = dt.types
    assert isinstance(returned_types, pd.DataFrame)
    assert 'Physical Type' in returned_types.columns
    assert 'Logical Type' in returned_types.columns
    assert 'Semantic Tag(s)' in returned_types.columns
    assert returned_types.shape[1] == 3
    assert len(returned_types.index) == len(sample_df.columns)
    assert all([col_dict['logical_type'] in ww.type_system.registered_types or isinstance(col_dict['logical_type'], LogicalType) for col_dict in dt.columns.values()])
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
    # --> shouldn't be in init
    dt = Schema(small_df)

    dt_repr = repr(dt)
    expected_repr = '                         Physical Type Logical Type Semantic Tag(s)\nColumn                                                             \nsample_datetime_series  datetime64[ns]     Datetime              []'
    assert dt_repr == expected_repr

    dt_html_repr = dt._repr_html_()
    expected_repr = '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Physical Type</th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>sample_datetime_series</th>\n      <td>datetime64[ns]</td>\n      <td>Datetime</td>\n      <td>[]</td>\n    </tr>\n  </tbody>\n</table>'
    assert dt_html_repr == expected_repr


def test_schema_repr_empty(empty_df):
    dt = Schema(empty_df)
    assert repr(dt) == 'Empty Schema'

    assert dt._repr_html_() == 'Empty Schema'


def test_schema_equality(sample_combos):
    # --> shouldn't be in init
    # --> commented out lines to be added back in when updating schema is allowed
    sample_df, sample_series = sample_combos
    dt_basic = Schema(sample_df)
    dt_basic2 = Schema(sample_df.copy())
    dt_names = Schema(sample_df, name='test')

    assert dt_basic != dt_names
    assert dt_basic == dt_basic2
    # dt_basic2.pop('id')
    # assert dt_basic != dt_basic2

    dt_index = Schema(sample_df, index='id')
    dt_time_index = Schema(sample_df, time_index='signup_date')
    # dt_set_index = dt_basic.set_index('id')

    assert dt_basic != dt_index
    # assert dt_index == dt_set_index
    assert dt_index != dt_time_index

    # # Check datatable with same parameters but changed underlying df
    # # We only check underlying data for equality with pandas dataframes
    # dt_set_index['phone_number'] = DataColumn(sample_series.rename('phone_number'), logical_type='NaturalLanguage')
    # if isinstance(dt_index.to_dataframe(), pd.DataFrame):
    #     assert dt_index != dt_set_index
    # else:
    #     assert dt_index == dt_set_index

    dt_numeric_time_index = Schema(sample_df, time_index='id')

    assert dt_time_index != dt_numeric_time_index

    dt_with_ltypes = Schema(sample_df, time_index='id', logical_types={'full_name': 'categorical'})
    assert dt_with_ltypes != dt_time_index
    # assert dt_with_ltypes == dt_numeric_time_index.set_types(logical_types={'full_name': Categorical})
    # assert dt_with_ltypes != dt_numeric_time_index.set_types(logical_types={'full_name': Categorical()})

    dt_with_metadata = Schema(sample_df, index='id', table_metadata={'created_by': 'user1'})
    assert Schema(sample_df, index='id') != dt_with_metadata
    assert Schema(sample_df, index='id', table_metadata={'created_by': 'user1'}) == dt_with_metadata
    assert Schema(sample_df, index='id', table_metadata={'created_by': 'user2'}) != dt_with_metadata
