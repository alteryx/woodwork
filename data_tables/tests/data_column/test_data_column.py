import pandas as pd

from data_tables.data_column import DataColumn, infer_logical_type
from data_tables.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    NaturalLanguage,
    Timedelta
)


def test_data_column_init(sample_series):
    data_col = DataColumn(sample_series)
    assert data_col.series is sample_series
    assert data_col.name == sample_series.name
    assert data_col.logical_type == Categorical
    assert data_col.tags == set()


def test_data_column_init_with_logical_type(sample_series):
    data_col = DataColumn(sample_series, NaturalLanguage)
    assert data_col.logical_type == NaturalLanguage


def test_infer_variable_types():
    df = pd.DataFrame({'id': [0, 1, 2],
                       'category': ['a', 'b', 'a'],
                       'ints': ['1', '2', '1'],
                       'boolean': [True, False, True],
                       'date_as_string': ['3/11/2000', '3/12/2000', '3/13/2000'],
                       'date_as_datetime': ['3/11/2000', '3/12/2000', '3/13/2000'],
                       'integers': [1, 2, 1],
                       'integers_category': [1, 2, 1],
                       'integers_object_dtype': [1, 2, 1],
                       'natural_language': ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown'],
                       'timedelta': pd.to_timedelta(range(3), unit='s')})

    df['date_as_datetime'] = pd.to_datetime(df['date_as_datetime'])
    df['integers_category'] = df['integers_category'].astype('category')
    df['integers_object_dtype'] = df['integers_object_dtype'].astype('object')

    correct_logical_types = {
        'id': Double,
        'category': Categorical,
        'ints': Categorical,
        'boolean': Boolean,
        'date_as_string': Datetime,
        'date_as_datetime': Datetime,
        'integers': Double,
        'integers_category': Categorical,
        'integers_object_dtype': Categorical,
        'natural_language': NaturalLanguage,
        'timedelta': Timedelta
    }

    for col in df.columns:
        inferred_type = infer_logical_type(df[col])
        assert inferred_type == correct_logical_types[col]
