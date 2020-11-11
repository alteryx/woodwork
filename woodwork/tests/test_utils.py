import os

import numpy as np
import pandas as pd
import pytest

import woodwork as ww
from woodwork.logical_types import (
    Categorical,
    Datetime,
    Double,
    LogicalType,
    str_to_logical_type
)
from woodwork.utils import (
    _convert_input_to_set,
    _get_mode,
    _get_specified_ltype_params,
    _is_numeric_series,
    _is_s3,
    _is_url,
    _new_dt_including,
    camel_to_snake,
    import_or_none,
    import_or_raise,
    list_logical_types,
    list_semantic_tags
)


def test_camel_to_snake():
    test_items = {
        'ZIPCode': 'zip_code',
        'SubRegionCode': 'sub_region_code',
        'NaturalLanguage': 'natural_language',
        'Categorical': 'categorical',
    }

    for key, value in test_items.items():
        assert camel_to_snake(key) == value


def test_convert_input_to_set():
    error_message = "semantic_tags must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _convert_input_to_set(int)

    error_message = "test_text must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _convert_input_to_set({'index': {}, 'time_index': {}}, 'test_text')

    error_message = "include parameter must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        _convert_input_to_set(['index', 1], 'include parameter')

    semantic_tags_from_single = _convert_input_to_set('index', 'include parameter')
    assert semantic_tags_from_single == {'index'}

    semantic_tags_from_list = _convert_input_to_set(['index', 'numeric', 'category'])
    assert semantic_tags_from_list == {'index', 'numeric', 'category'}

    semantic_tags_from_set = _convert_input_to_set({'index', 'numeric', 'category'}, 'include parameter')
    assert semantic_tags_from_set == {'index', 'numeric', 'category'}


def test_list_logical_types():
    all_ltypes = LogicalType.__subclasses__()

    df = list_logical_types()

    assert set(df.columns) == {'name', 'type_string', 'description', 'physical_type', 'standard_tags'}

    assert len(all_ltypes) == len(df)
    for name in df['name']:
        assert str_to_logical_type(name) in all_ltypes


def test_list_semantic_tags():
    df = list_semantic_tags()

    assert set(df.columns) == {'name', 'is_standard_tag', 'valid_logical_types'}

    for name, log_type_list in df[['name', 'valid_logical_types']].values:
        if name not in ['index', 'time_index', 'date_of_birth']:
            for log_type in log_type_list:
                assert name in log_type.standard_tags


def test_get_mode():
    series_list = [
        pd.Series([1, 2, 3, 4, 2, 2, 3]),
        pd.Series(['a', 'b', 'b', 'c', 'b']),
        pd.Series([3, 2, 3, 2]),
        pd.Series([np.nan, np.nan, np.nan]),
        pd.Series([pd.NA, pd.NA, pd.NA]),
        pd.Series([1, 2, np.nan, 2, np.nan, 3, 2]),
        pd.Series([1, 2, pd.NA, 2, pd.NA, 3, 2])
    ]

    answer_list = [2, 'b', 2, None, None, 2, 2]

    for series, answer in zip(series_list, answer_list):
        mode = _get_mode(series)
        if answer is None:
            assert mode is None
        else:
            assert mode == answer


def test_read_csv_no_params(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.csv')
    sample_df_pandas.to_csv(filepath, index=False)

    dt_from_csv = ww.read_csv(filepath=filepath)
    dt = ww.DataTable(sample_df_pandas)
    assert isinstance(dt, ww.DataTable)
    assert dt_from_csv.logical_types == dt.logical_types
    assert dt_from_csv.semantic_tags == dt.semantic_tags
    pd.testing.assert_frame_equal(dt_from_csv.to_dataframe(), dt.to_dataframe())


def test_read_csv_with_woodwork_params(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.csv')
    sample_df_pandas.to_csv(filepath, index=False)
    logical_types = {
        'full_name': 'NaturalLanguage',
        'phone_number': 'PhoneNumber'
    }
    semantic_tags = {
        'age': ['tag1', 'tag2'],
        'is_registered': ['tag3', 'tag4']
    }
    dt_from_csv = ww.read_csv(filepath=filepath,
                              index='id',
                              time_index='signup_date',
                              logical_types=logical_types,
                              semantic_tags=semantic_tags)
    dt = ww.DataTable(sample_df_pandas,
                      index='id',
                      time_index='signup_date',
                      logical_types=logical_types,
                      semantic_tags=semantic_tags)

    assert isinstance(dt, ww.DataTable)
    assert dt_from_csv.logical_types == dt.logical_types
    assert dt_from_csv.semantic_tags == dt.semantic_tags
    pd.testing.assert_frame_equal(dt_from_csv.to_dataframe(), dt.to_dataframe())


def test_read_csv_with_pandas_params(sample_df_pandas, tmpdir):
    filepath = os.path.join(tmpdir, 'sample.csv')
    sample_df_pandas.to_csv(filepath, index=False)
    nrows = 2
    dt_from_csv = ww.read_csv(filepath=filepath, nrows=nrows)
    dt = ww.DataTable(sample_df_pandas)

    assert isinstance(dt, ww.DataTable)
    assert dt_from_csv.logical_types == dt.logical_types
    assert dt_from_csv.semantic_tags == dt.semantic_tags
    assert len(dt_from_csv.to_dataframe()) == nrows
    pd.testing.assert_frame_equal(dt_from_csv.to_dataframe(), dt.to_dataframe().head(nrows))


def test_is_numeric_datetime_series(time_index_df):
    assert _is_numeric_series(time_index_df['ints'], None)
    assert _is_numeric_series(time_index_df['ints'], Double)
    assert not _is_numeric_series(time_index_df['ints'], Categorical)
    assert _is_numeric_series(time_index_df['ints'], Datetime)

    assert not _is_numeric_series(time_index_df['strs'], None)
    assert not _is_numeric_series(time_index_df['strs'], 'Categorical')
    assert not _is_numeric_series(time_index_df['strs'], Categorical)
    assert _is_numeric_series(time_index_df['strs'], Double)
    assert _is_numeric_series(time_index_df['strs'], 'Double')

    assert not _is_numeric_series(time_index_df['bools'], None)
    assert not _is_numeric_series(time_index_df['bools'], 'Boolean')

    assert not _is_numeric_series(time_index_df['times'], None)
    assert not _is_numeric_series(time_index_df['times'], Datetime)

    assert not _is_numeric_series(time_index_df['letters'], None)
    assert not _is_numeric_series(time_index_df['letters'], Double)
    assert not _is_numeric_series(time_index_df['letters'], Categorical)


def test_get_ltype_params():
    params_empty_class = _get_specified_ltype_params(Categorical)
    assert params_empty_class == {}
    params_empty = _get_specified_ltype_params(Categorical())
    assert params_empty == {}

    params_class = _get_specified_ltype_params(Datetime)
    assert params_class == {}

    params_null = _get_specified_ltype_params(Datetime())
    assert params_null == {'datetime_format': None}

    ymd = '%Y-%m-%d'
    params_value = _get_specified_ltype_params(Datetime(datetime_format=ymd))
    assert params_value == {'datetime_format': ymd}


def test_new_dt_including(sample_df_pandas):
    # more thorough testing for this exists in indexer testing and new_dt_from_cols testing
    dt = ww.DataTable(sample_df_pandas)
    new_dt = _new_dt_including(dt, sample_df_pandas.iloc[:, 1:4])
    for col in new_dt.columns:
        assert new_dt.semantic_tags[col] == new_dt.semantic_tags[col]
        assert new_dt.logical_types[col] == new_dt.logical_types[col]


def test_import_or_raise():
    assert import_or_raise('pandas', 'Module pandas could not be found') == pd

    error = 'Module nonexistent could not be found.'
    with pytest.raises(ImportError, match=error):
        import_or_raise('nonexistent', error)


def test_import_or_none():
    assert import_or_none('pandas') == pd
    assert import_or_none('nonexistent') is None


def test_is_url():
    assert _is_url('https://www.google.com/')
    assert not _is_url('google.com')


def test_is_s3():
    assert _is_s3('s3://test-bucket/test-key')
    assert not _is_s3('https://woodwork-static.s3.amazonaws.com/')
