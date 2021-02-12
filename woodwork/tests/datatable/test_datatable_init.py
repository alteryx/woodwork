import numpy as np
import pandas as pd
import pytest

from woodwork import DataTable
from woodwork.logical_types import (
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage
)
from woodwork.tests.testing_utils import to_pandas


def test_datatable_init(sample_df):
    dt = DataTable(sample_df)
    df = dt.to_dataframe()

    assert dt.name is None
    assert dt.index is None
    assert dt.time_index is None

    assert set(dt.columns.keys()) == set(sample_df.columns)
    assert df is sample_df
    pd.testing.assert_frame_equal(to_pandas(df), to_pandas(sample_df))


def test_datatable_init_with_name_and_index_vals(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   index='id',
                   time_index='signup_date')

    assert dt.name == 'datatable'
    assert dt.index == 'id'
    assert dt.time_index == 'signup_date'
    assert dt.columns[dt.time_index].logical_type == Datetime


def test_datatable_init_with_valid_string_time_index(time_index_df):
    dt = DataTable(time_index_df,
                   name='datatable',
                   index='id',
                   time_index='times')

    assert dt.name == 'datatable'
    assert dt.index == 'id'
    assert dt.time_index == 'times'
    assert dt.columns[dt.time_index].logical_type == Datetime


def test_datatable_init_with_invalid_string_time_index(sample_df):
    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        DataTable(sample_df, name='datatable', time_index='full_name')


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
        'age': 'Double'
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types=logical_types)
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == Double

    logical_types = {
        'full_name': 'NaturalLanguage',
        'age': 'Integer',
        'signup_date': 'Datetime'
    }
    dt = DataTable(sample_df,
                   name='datatable',
                   logical_types=logical_types,
                   time_index='signup_date')
    assert dt.columns['full_name'].logical_type == NaturalLanguage
    assert dt.columns['age'].logical_type == Integer
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


def test_datatable_init_with_numpy(sample_df_pandas):
    numpy_df = sample_df_pandas.to_numpy()

    dt = DataTable(numpy_df, index=0)
    assert set(dt.columns.keys()) == {i for i in range(len(numpy_df[0]))}
    assert dt.index == 0
    assert dt[0].logical_type == Categorical
    assert dt[1].logical_type == NaturalLanguage
    assert dt[5].logical_type == Datetime

    np_ints = np.array([[1, 0],
                        [2, 4],
                        [3, 6],
                        [4, 1]])
    dt = DataTable(np_ints)
    assert dt[0].logical_type == Integer
    assert dt[1].logical_type == Integer
    dt = dt.set_index(0)
    assert dt.index == 0

    dt = DataTable(np_ints, time_index=0, logical_types={0: 'Double', 1: Datetime}, semantic_tags={1: 'numeric_datetime'})
    assert dt.time_index == 0
    assert dt[0].logical_type == Double
    assert dt[0].semantic_tags == {'numeric', 'time_index'}
    assert dt[1].logical_type == Datetime
    assert dt[1].semantic_tags == {'numeric_datetime'}


def test_datatable_init_with_column_metadata(sample_df):
    column_metadata = {
        'age': {'interesting_values': [33]},
        'signup_date': {'description': 'date of account creation'}
    }
    dt = DataTable(sample_df, column_metadata=column_metadata)
    for name, column in dt.columns.items():
        assert column.metadata == (column_metadata.get(name) or {})
