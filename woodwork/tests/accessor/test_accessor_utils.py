import pytest

import woodwork as ww
from woodwork.accessor_utils import (
    _get_valid_dtype,
    _is_dataframe,
    _is_series,
    init_series
)
from woodwork.exceptions import TypeConversionError
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    NaturalLanguage
)
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_init_series_valid_conversion_specified_ltype(sample_series, use_both_dtypes):
    if ks and isinstance(sample_series, ks.Series):
        sample_series = sample_series.astype('str')
    else:
        if ww.config.get_option('use_nullable_dtypes'):
            sample_series = sample_series.astype('object')
        else:
            sample_series = sample_series.astype('string')

    series = init_series(sample_series, logical_type='categorical')
    assert series is not sample_series
    correct_dtype = _get_valid_dtype(type(sample_series), Categorical)
    assert series.dtype == correct_dtype
    assert series.ww.logical_type == Categorical
    assert series.ww.semantic_tags == {'category'}

    series = init_series(sample_series, logical_type='natural_language')
    assert series is not sample_series
    correct_dtype = _get_valid_dtype(type(sample_series), NaturalLanguage)
    assert series.dtype == correct_dtype
    assert series.ww.logical_type == NaturalLanguage
    assert series.ww.semantic_tags == set()


def test_init_series_valid_conversion_inferred_ltype(sample_series, use_both_dtypes):
    if ks and isinstance(sample_series, ks.Series):
        sample_series = sample_series.astype('str')
    else:
        sample_series = sample_series.astype('object')

    series = init_series(sample_series)
    assert series is not sample_series
    correct_dtype = _get_valid_dtype(type(sample_series), Categorical)
    assert series.dtype == correct_dtype
    assert series.ww.logical_type == Categorical
    assert series.ww.semantic_tags == {'category'}


def test_init_series_with_datetime(sample_datetime_series, use_both_dtypes):
    series = init_series(sample_datetime_series, logical_type='datetime')
    assert series.dtype == 'datetime64[ns]'
    assert series.ww.logical_type == Datetime


def test_init_series_all_parameters(sample_series, use_both_dtypes):
    if ks and isinstance(sample_series, ks.Series):
        sample_series = sample_series.astype('str')
    else:
        sample_series = sample_series.astype('object')

    metadata = {'meta_key': 'meta_value'}
    description = 'custom description'
    series = init_series(sample_series,
                         logical_type='categorical',
                         semantic_tags=['custom_tag'],
                         metadata=metadata,
                         description=description,
                         use_standard_tags=False)
    assert series is not sample_series
    correct_dtype = _get_valid_dtype(type(sample_series), Categorical)
    assert series.dtype == correct_dtype
    assert series.ww.logical_type == Categorical
    assert series.ww.semantic_tags == {'custom_tag'}
    assert series.ww.metadata == metadata
    assert series.ww.description == description


def test_init_series_error_on_invalid_conversion(sample_series, use_both_dtypes):
    if dd and isinstance(sample_series, dd.Series):
        pytest.xfail('Dask type conversion with astype does not fail until compute is called')
    if ks and isinstance(sample_series, ks.Series):
        pytest.xfail('Koalas allows this conversion, filling values it cannot convert with NaN '
                     'and converting dtype to float.')

    if ww.config.get_option('use_nullable_dtypes'):
        dtype = 'Int64'
    else:
        dtype = 'int64'
    error_message = f"Error converting datatype for sample_series from type category to type {dtype}. " \
        "Please confirm the underlying data is consistent with logical type Integer."
    with pytest.raises(TypeConversionError, match=error_message):
        init_series(sample_series, logical_type='integer')


def test_is_series(sample_df):
    assert _is_series(sample_df['id'])
    assert not _is_series(sample_df)


def test_is_dataframe(sample_df):
    assert _is_dataframe(sample_df)
    assert not _is_dataframe(sample_df['id'])


def test_get_valid_dtype(sample_series, use_both_dtypes):
    valid_dtype = _get_valid_dtype(type(sample_series), Categorical)
    if ks and isinstance(sample_series, ks.Series):
        assert valid_dtype == Categorical.backup_dtype
    else:
        assert valid_dtype == Categorical.primary_dtype

    valid_dtype = _get_valid_dtype(type(sample_series), Boolean)
    assert valid_dtype == Boolean.primary_dtype
