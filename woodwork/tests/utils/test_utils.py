
import re

import numpy as np
import pandas as pd
import pytest
from mock import patch

import woodwork as ww
from woodwork.logical_types import (
    Age,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    Integer,
    IntegerNullable,
    Ordinal,
    PostalCode,
    SubRegionCode
)
from woodwork.type_sys.type_system import DEFAULT_INFERENCE_FUNCTIONS
from woodwork.type_sys.utils import (
    _get_specified_ltype_params,
    _is_categorical_series,
    _is_numeric_series,
    col_is_datetime,
    list_logical_types,
    list_semantic_tags
)
from woodwork.utils import (
    _convert_input_to_set,
    _get_column_logical_type,
    _infer_datetime_format,
    _is_null_latlong,
    _is_s3,
    _is_url,
    _is_valid_latlong_series,
    _is_valid_latlong_value,
    _parse_logical_type,
    _reformat_to_latlong,
    _to_latlong_float,
    camel_to_snake,
    get_valid_mi_types,
    import_or_none,
    import_or_raise
)

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_camel_to_snake():
    test_items = {
        'PostalCode': 'postal_code',
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


@patch("woodwork.utils._validate_string_tags")
@patch("woodwork.utils._validate_tags_input_type")
def test_validation_methods_called(mock_validate_input_type, mock_validate_strings):
    assert not mock_validate_input_type.called
    assert not mock_validate_strings.called

    _convert_input_to_set('test_tag', validate=False)
    assert not mock_validate_input_type.called

    _convert_input_to_set('test_tag', validate=True)
    assert mock_validate_input_type.called

    _convert_input_to_set(['test_tag', 'tag2'], validate=False)
    assert not mock_validate_strings.called

    _convert_input_to_set(['test_tag', 'tag2'], validate=True)
    assert mock_validate_strings.called


def test_list_logical_types_default():
    all_ltypes = ww.logical_types.LogicalType.__subclasses__()

    df = list_logical_types()

    assert set(df.columns) == {'name', 'type_string', 'description', 'physical_type',
                               'standard_tags', 'is_default_type', 'is_registered', 'parent_type'}

    assert len(all_ltypes) == len(df)
    default_types_set = {str(cls) for cls in DEFAULT_INFERENCE_FUNCTIONS.keys()}
    listed_as_default = set(df[df['is_default_type']]['name'])
    assert listed_as_default == default_types_set


def test_list_logical_types_customized_type_system():
    ww.type_system.remove_type('URL')

    class CustomRegistered(ww.logical_types.LogicalType):
        primary_dtype = 'int64'

    class CustomNotRegistered(ww.logical_types.LogicalType):
        primary_dtype = 'int64'

    ww.type_system.add_type(CustomRegistered)
    all_ltypes = ww.logical_types.LogicalType.__subclasses__()
    df = list_logical_types()
    assert len(all_ltypes) == len(df)
    # Check that URL is unregistered
    url = df[df.name == 'URL'].iloc[0]
    assert url.is_default_type
    assert not url.is_registered

    # Check that new registered type is present and shows as registered
    index = df.name == 'CustomRegistered'
    assert index.any()
    custom = df[index].iloc[0]
    assert not custom.is_default_type
    assert custom.is_registered

    # Check that new unregistered type is present and shows as not registered
    index = df.name == 'CustomNotRegistered'
    assert index.any()
    custom = df[index].iloc[0]
    assert not custom.is_default_type
    assert not custom.is_registered
    ww.type_system.reset_defaults()


def test_list_semantic_tags():
    df = list_semantic_tags()

    assert set(df.columns) == {'name', 'is_standard_tag', 'valid_logical_types'}

    for name, log_type_list in df[['name', 'valid_logical_types']].values:
        if name not in ['index', 'time_index', 'date_of_birth']:
            for log_type in log_type_list:
                assert name in log_type.standard_tags


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


def test_reformat_to_latlong_errors():
    for latlong in [{1, 2, 3}, '{1, 2, 3}', 'This is text']:
        error = (f'LatLongs must either be a tuple, a list, or a string representation of a tuple. {latlong} does not fit the criteria.')
        with pytest.raises(ValueError, match=error):
            _reformat_to_latlong(latlong)

    error = re.escape("LatLongs must either be a tuple, a list, or a string representation of a tuple. (1,2) does not fit the criteria.")
    with pytest.raises(ValueError, match=error):
        _reformat_to_latlong("'(1,2)'")

    for latlong in [(1, 2, 3), '(1, 2, 3)']:
        error = re.escape("LatLong values must have exactly two values. (1, 2, 3) does not have two values.")
        with pytest.raises(ValueError, match=error):
            _reformat_to_latlong(latlong)

    error = re.escape("Latitude and Longitude values must be in decimal degrees. The latitude or longitude represented by 41deg52\'54\" N cannot be converted to a float.")
    with pytest.raises(ValueError, match=error):
        _reformat_to_latlong(('41deg52\'54\" N', '21deg22\'54\" W'))


def test_reformat_to_latlong():
    simple_latlong = (1, 2)

    assert _reformat_to_latlong((1, 2)) == simple_latlong
    assert _reformat_to_latlong(('1', '2')) == simple_latlong
    assert _reformat_to_latlong('(1,2)') == simple_latlong

    # Check non-standard tuple formats
    assert _reformat_to_latlong([1, 2]) == simple_latlong
    assert _reformat_to_latlong(['1', '2']) == simple_latlong
    assert _reformat_to_latlong('[1, 2]') == simple_latlong
    assert _reformat_to_latlong('1, 2') == simple_latlong

    assert _reformat_to_latlong(None) is np.nan
    assert _reformat_to_latlong((1, np.nan)) == (1, np.nan)
    assert _reformat_to_latlong((np.nan, '1')) == (np.nan, 1)

    # This is how csv and parquet will deserialize
    assert _reformat_to_latlong('(1, nan)') == (1, np.nan)
    assert _reformat_to_latlong('(NaN, 9)') == (np.nan, 9)


def test_reformat_to_latlong_list():
    simple_latlong = [1, 2]

    assert _reformat_to_latlong((1, 2), use_list=True) == simple_latlong
    assert _reformat_to_latlong(('1', '2'), use_list=True) == simple_latlong
    assert _reformat_to_latlong('(1,2)', use_list=True) == simple_latlong

    assert _reformat_to_latlong([1, 2], use_list=True) == simple_latlong
    assert _reformat_to_latlong(['1', '2'], use_list=True) == simple_latlong
    assert _reformat_to_latlong('[1, 2]', use_list=True) == simple_latlong
    assert _reformat_to_latlong('1, 2', use_list=True) == simple_latlong

    assert _reformat_to_latlong((1, np.nan), use_list=True) == [1, np.nan]
    assert _reformat_to_latlong((np.nan, '1'), use_list=True) == [np.nan, 1]

    # This is how csv and parquet will deserialize
    assert _reformat_to_latlong('[1, nan]', use_list=True) == [1, np.nan]
    assert _reformat_to_latlong('[1, NaN]', use_list=True) == [1, np.nan]


def test_to_latlong_float():
    assert _to_latlong_float(4) == 4.0
    assert _to_latlong_float('2.2') == 2.2

    assert _to_latlong_float(None) is np.nan
    assert _to_latlong_float(np.nan) is np.nan
    assert _to_latlong_float(pd.NA) is np.nan

    error = re.escape('Latitude and Longitude values must be in decimal degrees. The latitude or longitude represented by [1, 2, 3] cannot be converted to a float.')
    with pytest.raises(ValueError, match=error):
        _to_latlong_float([1, 2, 3])


def test_is_null_latlong():
    assert _is_null_latlong(None)
    assert _is_null_latlong(np.nan)
    assert _is_null_latlong(pd.NA)
    assert _is_null_latlong('None')
    assert _is_null_latlong('nan')
    assert _is_null_latlong('NaN')

    assert not _is_null_latlong([None, 1, 3])
    assert not _is_null_latlong('none')
    assert not _is_null_latlong(0)
    assert not _is_null_latlong(False)


def test_is_valid_latlong_value():
    values = [
        (1.0, 2.0),
        np.nan,
        [1.0, 2.0],
        (np.nan, np.nan),
        ('a', 2.0),
        (1.0, 2.0, 3.0),
        None
    ]

    expected_values = [
        True,
        True,
        False,
        False,
        False,
        False,
        False
    ]

    for index, value in enumerate(values):
        assert _is_valid_latlong_value(value) is expected_values[index]


def test_is_valid_latlong_value_koalas():
    values = [
        (1.0, 2.0),
        np.nan,
        [1.0, 2.0],
        (np.nan, np.nan),
        ('a', 2.0),
        (1.0, 2.0, 3.0),
        None
    ]

    expected_values = [
        False,
        True,
        True,
        False,
        False,
        False,
        False
    ]

    for index, value in enumerate(values):
        assert _is_valid_latlong_value(value, bracket_type=list) is expected_values[index]


def test_is_valid_latlong_series():
    valid_series = pd.Series([(1.0, 2.0), (3.0, 4.0)])
    invalid_series = pd.Series([(1.0, 2.0), (3.0, '4.0')])

    assert _is_valid_latlong_series(valid_series) is True
    assert _is_valid_latlong_series(invalid_series) is False


def test_get_valid_mi_types():
    valid_types = get_valid_mi_types()
    expected_types = [
        Age,
        AgeNullable,
        Boolean,
        BooleanNullable,
        Categorical,
        CountryCode,
        Datetime,
        Double,
        Integer,
        IntegerNullable,
        Ordinal,
        PostalCode,
        SubRegionCode,
    ]

    assert valid_types == expected_types


def test_get_column_logical_type(sample_series):
    assert isinstance(_get_column_logical_type(sample_series, None, 'col_name'), Categorical)

    assert isinstance(_get_column_logical_type(sample_series, Datetime, 'col_name'), Datetime)


def test_parse_logical_type():
    assert isinstance(_parse_logical_type('Datetime', 'col_name'), Datetime)
    assert isinstance(_parse_logical_type(Datetime, 'col_name'), Datetime)

    ymd_format = Datetime(datetime_format='%Y-%m-%d')
    assert _parse_logical_type(ymd_format, 'col_name') == ymd_format


def test_parse_logical_type_errors():
    error = "Invalid logical type specified for 'col_name'"
    with pytest.raises(TypeError, match=error):
        _parse_logical_type(int, 'col_name')


def test_col_is_datetime():
    inputs = [
        pd.to_datetime(pd.Series(['2020-01-01', '2021-02-02', '2022-03-03'])),
        pd.to_datetime(pd.Series([pd.NA, '2021-02-02', '2022-03-03'])),
        pd.Series([1, 2, 3]),
        pd.Series([pd.NA, 2, 3]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([pd.NA, 2.0, 3.0]),
        pd.Series(['2020-01-01', '2021-02-02', '2022-03-03']),
        pd.Series([pd.NA, '2021-02-02', '2022-03-03']),
        pd.Series(['a', 'b', 'c']),
        pd.Series([pd.NA, 'b', 'c']),
        pd.Series([pd.NA, pd.NA, pd.NA]),
    ]

    expected_values = [
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]

    for input, expected in list(zip(inputs, expected_values)):
        actual = col_is_datetime(input)
        assert actual is expected


def test_infer_datetime_format(datetimes):
    for series in datetimes:
        fmt = _infer_datetime_format(series)
        assert fmt == '%m/%d/%Y'

    dt1 = pd.Series(['3/11/2000 9:00', '3/11/2000 10:00', '3/11/2000 11:00', '3/11/2000 12:00'])
    fmt = _infer_datetime_format(dt1)
    assert fmt == '%m/%d/%Y %H:%M'


def test_infer_datetime_format_all_null():
    missing_data = [
        pd.Series([None, None, None]),
        pd.Series([np.nan, np.nan, np.nan]),
        pd.Series([pd.NA, pd.NA, pd.NA]),
        pd.Series([]),
    ]

    for pd_series in missing_data:
        assert _infer_datetime_format(pd_series) is None
        if dd:
            dd_series = dd.from_pandas(pd_series, npartitions=2)
            assert _infer_datetime_format(dd_series) is None
        if ks:
            ks_series = ks.from_pandas(pd_series)
            assert _infer_datetime_format(ks_series) is None


def test_is_categorical() -> None:
    # not categorical because unhashable type (list)
    assert not _is_categorical_series(pd.Series([[1]]), 0)
    assert not _is_categorical_series(pd.Series([None, [1]]), 0)

    # not categorical because empty series
    assert not _is_categorical_series(pd.Series([]), 0)
    assert not _is_categorical_series(pd.Series([None]), 0)
    assert not _is_categorical_series(pd.Series([None, None]), 0)

    # not categorical because too many unique values
    assert not _is_categorical_series(pd.Series([1, 2]), 0.5)
    assert not _is_categorical_series(pd.Series([1, 2, 3, 1]), 0.5)
    assert not _is_categorical_series(pd.Series([1, 2, 3, 4]), 0.75)

    # categorical
    assert _is_categorical_series(pd.Series([1, 1]), 0.5)
    assert _is_categorical_series(pd.Series([1, 2, 1, 1]), 0.5)
    assert _is_categorical_series(pd.Series([1, 2, 3, 1]), 0.75)
