import re

import numpy as np
import pandas as pd
import pytest

from woodwork.data_column import DataColumn, infer_logical_type
from woodwork.logical_types import (
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
    Ordinal,
    SubRegionCode,
    Timedelta,
    WholeNumber,
    ZIPCode
)


def test_data_column_init(sample_series):
    data_col = DataColumn(sample_series, add_standard_tags=False)
    assert data_col.series is sample_series
    assert data_col.name == sample_series.name
    assert data_col.logical_type == Categorical
    assert data_col.semantic_tags == set()


def test_data_column_init_with_logical_type(sample_series):
    data_col = DataColumn(sample_series, NaturalLanguage)
    assert data_col.logical_type == NaturalLanguage
    assert data_col.semantic_tags == set()

    data_col = DataColumn(sample_series, "natural_language")
    assert data_col.logical_type == NaturalLanguage
    assert data_col.semantic_tags == set()

    data_col = DataColumn(sample_series, "NaturalLanguage")
    assert data_col.logical_type == NaturalLanguage
    assert data_col.semantic_tags == set()


def test_data_column_init_with_semantic_tags(sample_series):
    semantic_tags = ['tag1', 'tag2']
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, add_standard_tags=False)
    assert data_col.semantic_tags == set(semantic_tags)


def test_data_column_with_alternate_semantic_tags_input(sample_series):
    semantic_tags = 'custom_tag'
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, add_standard_tags=False)
    assert data_col.semantic_tags == {'custom_tag'}

    semantic_tags = {'custom_tag', 'numeric'}
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, add_standard_tags=False)
    assert data_col.semantic_tags == semantic_tags


def test_invalid_logical_type(sample_series):
    error_message = "Invalid logical type specified for 'sample_series'"
    with pytest.raises(TypeError, match=error_message):
        DataColumn(sample_series, int)

    error_message = "String naturalllanguage is not a valid logical type"
    with pytest.raises(ValueError, match=error_message):
        DataColumn(sample_series, 'naturalllanguage')


def test_semantic_tag_errors(sample_series):
    error_message = "semantic_tags must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        DataColumn(sample_series, semantic_tags=int)

    error_message = "semantic_tags must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        DataColumn(sample_series, semantic_tags={'index': {}, 'time_index': {}})

    error_message = "semantic_tags must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        DataColumn(sample_series, semantic_tags=['index', 1])


def test_integer_inference():
    series_list = [
        pd.Series([-1, 2, 1]),
        pd.Series([-1, 0, 5]),
    ]
    dtypes = ['int8', 'int16', 'int32', 'int64', 'intp', 'int']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Integer


def test_whole_number_inference():
    series_list = [
        pd.Series([0, 1, 5]),
        pd.Series([2, 3, 5]),
    ]
    dtypes = ['int8', 'int16', 'int32', 'int64', 'uint8',
              'uint16', 'uint32', 'uint64', 'intp', 'uintp', 'int']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == WholeNumber


def test_double_inference():
    series_list = [
        pd.Series([-1, 2.0, 1]),
        pd.Series([1, np.nan, 1])
    ]
    dtypes = ['float', 'float32', 'float64', 'float_']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Double


def test_boolean_inference():
    series_list = [
        pd.Series([True, False, True]),
        pd.Series([True, False, np.nan]),
    ]
    dtypes = ['bool']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Boolean


def test_datetime_inference():
    series_list = [
        pd.Series(['3/11/2000', '3/12/2000', '3/13/2000']),
        pd.Series(['3/11/2000', '3/12/2000', np.nan]),
    ]
    dtypes = ['object', 'string', 'datetime64[ns]']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Datetime


def test_categorical_inference():
    series_list = [
        pd.Series(['a', 'b', 'a']),
        pd.Series(['1', '2', '1']),
        pd.Series(['a', 'b', np.nan]),
        pd.Series([1, 2, 1])
    ]
    dtypes = ['object', 'string', 'category']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Categorical


def test_timedelta_inference():
    series_list = [
        pd.Series(pd.to_timedelta(range(3), unit='s')),
        pd.Series([pd.to_timedelta(1, unit='s'), np.nan])
    ]
    dtypes = ['timedelta64[ns]']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == Timedelta


def test_natural_language_inference():
    series_list = [
        pd.Series(['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown']),
    ]
    dtypes = ['object', 'string']
    for series in series_list:
        for dtype in dtypes:
            inferred_type = infer_logical_type(series.astype(dtype))
            assert inferred_type == NaturalLanguage


def test_data_column_repr(sample_series):
    data_col = DataColumn(sample_series, add_standard_tags=False)
    assert data_col.__repr__() == '<DataColumn: sample_series (Physical Type = object) ' \
        '(Logical Type = Categorical) (Semantic Tags = set())>'


def test_set_semantic_tags(sample_series):
    semantic_tags = {'tag1', 'tag2'}
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, add_standard_tags=False)
    assert data_col.semantic_tags == semantic_tags

    new_tags = ['new_tag']
    data_col.set_semantic_tags(new_tags)
    assert data_col.semantic_tags == set(new_tags)


def test_set_semantic_tags_with_index(sample_series):
    semantic_tags = {'tag1', 'tag2'}
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, add_standard_tags=False)
    data_col._set_as_index()
    assert data_col.semantic_tags == {'tag1', 'tag2', 'index'}
    new_tags = ['new_tag']
    data_col.set_semantic_tags(new_tags)
    assert data_col.semantic_tags == {'index', 'new_tag'}
    data_col.set_semantic_tags(new_tags, retain_index_tags=False)
    assert data_col.semantic_tags == {'new_tag'}


def test_set_semantic_tags_with_time_index(sample_datetime_series):
    semantic_tags = {'tag1', 'tag2'}
    data_col = DataColumn(sample_datetime_series, semantic_tags=semantic_tags, add_standard_tags=False)
    data_col._set_as_time_index()
    assert data_col.semantic_tags == {'tag1', 'tag2', 'time_index'}
    new_tags = ['new_tag']
    data_col.set_semantic_tags(new_tags)
    assert data_col.semantic_tags == {'time_index', 'new_tag'}
    data_col.set_semantic_tags(new_tags, retain_index_tags=False)
    assert data_col.semantic_tags == {'new_tag'}


def test_adds_numeric_standard_tag():
    series = pd.Series([1, 2, 3])
    semantic_tags = 'custom_tag'

    logical_types = [Integer, Double, WholeNumber]
    for logical_type in logical_types:
        data_col = DataColumn(series, logical_type=logical_type, semantic_tags=semantic_tags)
        assert data_col.semantic_tags == {'custom_tag', 'numeric'}


def test_adds_category_standard_tag():
    series = pd.Series([1, 2, 3])
    semantic_tags = 'custom_tag'

    logical_types = [Categorical, CountryCode, Ordinal, SubRegionCode, ZIPCode]
    for logical_type in logical_types:
        data_col = DataColumn(series, logical_type=logical_type, semantic_tags=semantic_tags)
        assert data_col.semantic_tags == {'custom_tag', 'category'}


def test_does_not_add_standard_tags():
    series = pd.Series([1, 2, 3])
    semantic_tags = 'custom_tag'
    data_col = DataColumn(series,
                          logical_type=Double,
                          semantic_tags=semantic_tags,
                          add_standard_tags=False)
    assert data_col.semantic_tags == {'custom_tag'}


def test_add_custom_tags(sample_series):
    semantic_tags = 'initial_tag'
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, add_standard_tags=False)

    data_col.add_semantic_tags('string_tag')
    assert data_col.semantic_tags == {'initial_tag', 'string_tag'}

    data_col.add_semantic_tags(['list_tag'])
    assert data_col.semantic_tags == {'initial_tag', 'string_tag', 'list_tag'}

    data_col.add_semantic_tags({'set_tag'})
    assert data_col.semantic_tags == {'initial_tag', 'string_tag', 'list_tag', 'set_tag'}


def test_warns_on_setting_duplicate_tag(sample_series):
    semantic_tags = ['first_tag', 'second_tag']
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, add_standard_tags=False)

    expected_message = "Semantic tag(s) 'first_tag, second_tag' already present on column 'sample_series'"
    with pytest.warns(UserWarning) as record:
        data_col.add_semantic_tags(['first_tag', 'second_tag'])
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message


def test_set_logical_type_with_standard_tags(sample_series):
    data_col = DataColumn(sample_series,
                          logical_type=NaturalLanguage,
                          semantic_tags='original_tag',
                          add_standard_tags=True)

    new_col = data_col.set_logical_type(Categorical)
    assert isinstance(new_col, DataColumn)
    assert new_col is not data_col
    assert new_col.logical_type == Categorical
    assert new_col.semantic_tags == {'category'}


def test_set_logical_type_without_standard_tags(sample_series):
    data_col = DataColumn(sample_series,
                          logical_type=NaturalLanguage,
                          semantic_tags='original_tag',
                          add_standard_tags=False)

    new_col = data_col.set_logical_type(Categorical)
    assert isinstance(new_col, DataColumn)
    assert new_col is not data_col
    assert new_col.logical_type == Categorical
    assert new_col.semantic_tags == set()


def test_set_logical_type_retains_index_tag(sample_series):
    data_col = DataColumn(sample_series,
                          logical_type=NaturalLanguage,
                          semantic_tags='original_tag',
                          add_standard_tags=False)

    data_col._set_as_index()
    assert data_col.semantic_tags == {'index', 'original_tag'}
    new_col = data_col.set_logical_type(Categorical)
    assert new_col.semantic_tags == {'index'}
    new_col = data_col.set_logical_type(Categorical, retain_index_tags=False)
    assert new_col.semantic_tags == set()


def test_set_logical_type_retains_time_index_tag(sample_datetime_series):
    data_col = DataColumn(sample_datetime_series,
                          logical_type=Datetime,
                          semantic_tags='original_tag',
                          add_standard_tags=False)

    data_col._set_as_time_index()
    assert data_col.semantic_tags == {'time_index', 'original_tag'}
    new_col = data_col.set_logical_type(Categorical)
    assert new_col.semantic_tags == {'time_index'}
    new_col = data_col.set_logical_type(Categorical, retain_index_tags=False)
    assert new_col.semantic_tags == set()


def test_reset_semantic_tags_with_standard_tags(sample_series):
    semantic_tags = 'initial_tag'
    data_col = DataColumn(sample_series,
                          semantic_tags=semantic_tags,
                          logical_type=Categorical,
                          add_standard_tags=True)

    new_col = data_col.reset_semantic_tags()
    assert new_col is not data_col
    assert new_col.semantic_tags == Categorical.standard_tags


def test_reset_semantic_tags_without_standard_tags(sample_series):
    semantic_tags = 'initial_tag'
    data_col = DataColumn(sample_series,
                          semantic_tags=semantic_tags,
                          add_standard_tags=False)

    new_col = data_col.reset_semantic_tags()
    assert new_col is not data_col
    assert new_col.semantic_tags == set()


def test_reset_semantic_tags_with_index(sample_series):
    semantic_tags = 'initial_tag'
    data_col = DataColumn(sample_series,
                          semantic_tags=semantic_tags,
                          add_standard_tags=False)

    data_col._set_as_index()
    new_col = data_col.reset_semantic_tags(retain_index_tags=True)
    assert new_col.semantic_tags == {'index'}
    new_col = data_col.reset_semantic_tags()
    assert new_col.semantic_tags == set()


def test_reset_semantic_tags_with_time_index(sample_datetime_series):
    semantic_tags = 'initial_tag'
    data_col = DataColumn(sample_datetime_series,
                          semantic_tags=semantic_tags,
                          add_standard_tags=False)

    data_col._set_as_time_index()
    new_col = data_col.reset_semantic_tags(retain_index_tags=True)
    assert new_col.semantic_tags == {'time_index'}
    new_col = data_col.reset_semantic_tags()
    assert new_col.semantic_tags == set()


def test_remove_semantic_tags(sample_series):
    tags_to_remove = [
        'tag1',
        ['tag1'],
        {'tag1'}
    ]

    data_col = DataColumn(sample_series,
                          semantic_tags=['tag1', 'tag2'],
                          add_standard_tags=False)

    for tag in tags_to_remove:
        new_col = data_col.remove_semantic_tags(tag)
        assert new_col is not data_col
        assert new_col.semantic_tags == {'tag2'}


def test_remove_standard_semantic_tag(sample_series):
    # Check that warning is raised if add_standard_tags is True - tag should be removed
    data_col = DataColumn(sample_series,
                          logical_type=Categorical,
                          semantic_tags='tag1',
                          add_standard_tags=True)
    expected_message = "Removing standard semantic tag(s) 'category' from column 'sample_series'"
    with pytest.warns(UserWarning) as record:
        new_col = data_col.remove_semantic_tags(['tag1', 'category'])
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message
    assert new_col.semantic_tags == set()

    # Check that warning is not raised if add_standard_tags is False - tag should be removed
    data_col = DataColumn(sample_series,
                          logical_type=Categorical,
                          semantic_tags=['category', 'tag1'],
                          add_standard_tags=False)

    with pytest.warns(None) as record:
        new_col = data_col.remove_semantic_tags(['tag1', 'category'])
    assert len(record) == 0
    assert new_col.semantic_tags == set()


def test_remove_semantic_tags_raises_error_with_invalid_tag(sample_series):
    data_col = DataColumn(sample_series,
                          semantic_tags='tag1')
    error_msg = re.escape("Semantic tag(s) 'invalid_tagname' not present on column 'sample_series'")
    with pytest.raises(LookupError, match=error_msg):
        data_col.remove_semantic_tags('invalid_tagname')


def test_raises_error_setting_index_tag_directly(sample_series):
    error_msg = re.escape("Cannot add 'index' tag directly. To set a column as the index, "
                          "use DataTable.set_index() instead.")
    with pytest.raises(ValueError, match=error_msg):
        DataColumn(sample_series, semantic_tags='index')

    data_col = DataColumn(sample_series)
    with pytest.raises(ValueError, match=error_msg):
        data_col.add_semantic_tags('index')
    with pytest.raises(ValueError, match=error_msg):
        data_col.set_semantic_tags('index')


def test_raises_error_setting_time_index_tag_directly(sample_series):
    error_msg = re.escape("Cannot add 'time_index' tag directly. To set a column as the time index, "
                          "use DataTable.set_time_index() instead.")
    with pytest.raises(ValueError, match=error_msg):
        DataColumn(sample_series, semantic_tags='time_index')

    data_col = DataColumn(sample_series)
    with pytest.raises(ValueError, match=error_msg):
        data_col.add_semantic_tags('time_index')
    with pytest.raises(ValueError, match=error_msg):
        data_col.set_semantic_tags('time_index')


def test_set_as_index(sample_series):
    data_col = DataColumn(sample_series)
    data_col._set_as_index()
    assert 'index' in data_col.semantic_tags


def test_set_as_time_index(sample_series):
    data_col = DataColumn(sample_series)
    data_col._set_as_time_index()
    assert 'time_index' in data_col.semantic_tags
