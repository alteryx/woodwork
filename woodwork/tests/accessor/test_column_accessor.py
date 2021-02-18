import re

import numpy as np
import pandas as pd
import pytest

from woodwork.exceptions import (
    DuplicateTagsWarning,
    TypeConversionError,
    TypingInfoMismatchWarning
)
from woodwork.logical_types import (
    Categorical,
    CountryCode,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
    Ordinal,
    SubRegionCode,
    ZIPCode
)
from woodwork.tests.testing_utils import xfail_dask_and_koalas
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_accessor_init(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    assert series.ww._schema is None
    series.ww.init()
    assert isinstance(series.ww._schema, dict)
    assert series.ww.logical_type == Categorical
    assert series.ww.semantic_tags == {'category'}


def test_accessor_init_with_logical_type(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.copy().astype('string')
    series.ww.init(logical_type=NaturalLanguage)
    assert series.ww.logical_type == NaturalLanguage
    assert series.ww.semantic_tags == set()

    series = sample_series.copy().astype('string')
    series.ww.init(logical_type="natural_language")
    assert series.ww.logical_type == NaturalLanguage
    assert series.ww.semantic_tags == set()

    series = sample_series.copy().astype('string')
    series.ww.init(logical_type="NaturalLanguage")
    assert series.ww.logical_type == NaturalLanguage
    assert series.ww.semantic_tags == set()


def test_accessor_init_with_invalid_logical_type(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series
    error_message = "Cannot initialize Woodwork. Series dtype 'object' is incompatible with " \
        "NaturalLanguage dtype. Try converting series dtype to 'string' before initializing " \
        "or use the woodwork.init_series function to initialize."
    with pytest.raises(ValueError, match=error_message):
        series.ww.init(logical_type=NaturalLanguage)


def test_accessor_init_with_semantic_tags(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    semantic_tags = ['tag1', 'tag2']
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)
    assert series.ww.semantic_tags == set(semantic_tags)


def test_accessor_warnings_accessing_properties_before_init(sample_series):
    xfail_dask_and_koalas(sample_series)
    error_message = "Woodwork not initialized for this Series. Initialize by calling Series.ww.init"

    with pytest.raises(AttributeError, match=error_message):
        sample_series.ww.__repr__()

    with pytest.raises(AttributeError, match=error_message):
        sample_series.ww.description

    with pytest.raises(AttributeError, match=error_message):
        sample_series.ww.logical_type

    with pytest.raises(AttributeError, match=error_message):
        sample_series.ww.metadata

    with pytest.raises(AttributeError, match=error_message):
        sample_series.ww.semantic_tags


# def test_datacolumn_init_with_extension_array():
#     series_categories = pd.Series([1, 2, 3], dtype='category')
#     extension_categories = pd.Categorical([1, 2, 3])

#     data_col = DataColumn(extension_categories)
#     series = data_col.to_series()
#     assert series.equals(series_categories)
#     assert series.name is None
#     assert data_col.name is None
#     assert data_col.dtype == 'category'
#     assert data_col.logical_type == Categorical

#     series_ints = pd.Series([1, 2, None, 4], dtype='Int64')
#     extension_ints = pd.arrays.IntegerArray(np.array([1, 2, 3, 4], dtype="int64"), mask=np.array([False, False, True, False]))

#     data_col_with_name = DataColumn(extension_ints, name='extension')
#     series = data_col_with_name.to_series()
#     assert series.equals(series_ints)
#     assert series.name == 'extension'
#     assert data_col_with_name.name == 'extension'

#     series_strs = pd.Series([1, 2, None, 4], dtype='string')

#     data_col_different_ltype = DataColumn(extension_ints, logical_type='NaturalLanguage')
#     series = data_col_different_ltype.to_series()
#     assert series.equals(series_strs)
#     assert data_col_different_ltype.logical_type == NaturalLanguage
#     assert data_col_different_ltype.dtype == 'string'


def test_accessor_with_alternate_semantic_tags_input(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.copy().astype('category')
    semantic_tags = 'custom_tag'
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)
    assert series.ww.semantic_tags == {'custom_tag'}

    series = sample_series.copy().astype('category')
    semantic_tags = {'custom_tag', 'numeric'}
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)
    assert series.ww.semantic_tags == semantic_tags


def test_logical_type_errors(sample_series):
    xfail_dask_and_koalas(sample_series)
    error_message = "Invalid logical type specified for 'sample_series'"
    with pytest.raises(TypeError, match=error_message):
        sample_series.ww.init(logical_type=int)

    error_message = "String naturalllanguage is not a valid logical type"
    with pytest.raises(ValueError, match=error_message):
        sample_series.ww.init(logical_type='naturalllanguage')


def test_semantic_tag_errors(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    error_message = "semantic_tags for sample_series must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        series.ww.init(semantic_tags=int)

    error_message = "semantic_tags for sample_series must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        series.ww.init(semantic_tags={'index': {}, 'time_index': {}})

    error_message = "semantic_tags for sample_series must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        series.ww.init(semantic_tags=['index', 1])


def test_accessor_description(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.copy().astype('category')

    column_description = "custom description"
    series.ww.init(description=column_description)
    assert series.ww.description == column_description

    new_description = "updated description text"
    series.ww.description = new_description
    assert series.ww.description == new_description


def test_description_error_on_init(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    err_msg = "Column description must be a string"
    with pytest.raises(TypeError, match=err_msg):
        series.ww.init(description=123)


def test_description_error_on_update(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init()
    err_msg = "Column description must be a string"
    with pytest.raises(TypeError, match=err_msg):
        series.ww.description = 123


def test_accessor_repr(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init(use_standard_tags=False)
    # Koalas doesn't support categorical
    if ks and isinstance(series, ks.Series):
        dtype = 'object'
    else:
        dtype = 'category'
    assert series.ww.__repr__() == f'<Series: sample_series (Physical Type = {dtype}) ' \
        '(Logical Type = Categorical) (Semantic Tags = set())>'


def test_set_semantic_tags(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    semantic_tags = {'tag1', 'tag2'}
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)
    assert series.ww.semantic_tags == semantic_tags

    new_tags = ['new_tag']
    series.ww.set_semantic_tags(new_tags)
    assert series.ww.semantic_tags == set(new_tags)


def test_set_semantic_tags_with_standard_tags(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    semantic_tags = {'tag1', 'tag2'}
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=True)
    assert series.ww.semantic_tags == semantic_tags.union({'category'})

    new_tags = ['new_tag']
    series.ww.set_semantic_tags(new_tags)
    assert series.ww.semantic_tags == set(new_tags).union({'category'})


def test_adds_numeric_standard_tag():
    series = pd.Series([1, 2, 3])
    semantic_tags = 'custom_tag'

    logical_types = [Integer, Double]
    for logical_type in logical_types:
        series = series.astype(logical_type.pandas_dtype)
        series.ww.init(logical_type=logical_type, semantic_tags=semantic_tags)
        assert series.ww.semantic_tags == {'custom_tag', 'numeric'}


def test_adds_category_standard_tag():
    semantic_tags = 'custom_tag'

    logical_types = [Categorical, CountryCode, Ordinal(order=(1, 2, 3)), SubRegionCode, ZIPCode]
    for logical_type in logical_types:
        series = pd.Series([1, 2, 3], dtype='category')
        series.ww.init(logical_type=logical_type, semantic_tags=semantic_tags)
        assert series.ww.semantic_tags == {'custom_tag', 'category'}


def test_does_not_add_standard_tags():
    series = pd.Series([1.1, 2, 3])
    semantic_tags = 'custom_tag'
    series.ww.init(logical_type=Double,
                   semantic_tags=semantic_tags,
                   use_standard_tags=False)
    assert series.ww.semantic_tags == {'custom_tag'}


def test_add_custom_tags(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    semantic_tags = 'initial_tag'
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)

    series.ww.add_semantic_tags('string_tag')
    assert series.ww.semantic_tags == {'initial_tag', 'string_tag'}

    series.ww.add_semantic_tags(['list_tag'])
    assert series.ww.semantic_tags == {'initial_tag', 'string_tag', 'list_tag'}

    series.ww.add_semantic_tags({'set_tag'})
    assert series.ww.semantic_tags == {'initial_tag', 'string_tag', 'list_tag', 'set_tag'}


def test_warns_on_setting_duplicate_tag(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    semantic_tags = ['first_tag', 'second_tag']
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)

    expected_message = "Semantic tag(s) 'first_tag, second_tag' already present on column 'sample_series'"
    with pytest.warns(DuplicateTagsWarning) as record:
        series.ww.add_semantic_tags(['first_tag', 'second_tag'])
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message


def test_set_logical_type_with_standard_tags(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')

    series.ww.init(logical_type='Categorical',
                   semantic_tags='original_tag',
                   use_standard_tags=True)

    new_series = series.ww.set_logical_type('CountryCode')
    assert series.ww.logical_type == Categorical
    assert new_series.ww.logical_type == CountryCode
    assert new_series.ww.semantic_tags == {'category'}


def test_set_logical_type_without_standard_tags(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')

    series.ww.init(logical_type='Categorical',
                   semantic_tags='original_tag',
                   use_standard_tags=False)

    new_series = series.ww.set_logical_type('CountryCode')
    assert series.ww.logical_type == Categorical
    assert new_series.ww.logical_type == CountryCode
    assert new_series.ww.semantic_tags == set()


def test_set_logical_type_valid_dtype_change(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init(logical_type='Categorical')

    new_series = series.ww.set_logical_type('NaturalLanguage')
    assert series.ww.logical_type == Categorical
    assert series.dtype == 'category'
    assert new_series.ww.logical_type == NaturalLanguage
    assert new_series.dtype == 'string'


def test_set_logical_type_invalid_dtype_change(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init(logical_type='Categorical')
    error_message = "Error converting datatype for sample_series from type category to " \
        "type Int64. Please confirm the underlying data is consistent with logical type Integer."
    with pytest.raises(TypeConversionError, match=error_message):
        series.ww.set_logical_type('Integer')


def test_reset_semantic_tags_with_standard_tags(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    semantic_tags = 'initial_tag'
    series.ww.init(semantic_tags=semantic_tags,
                   logical_type=Categorical,
                   use_standard_tags=True)

    series.ww.reset_semantic_tags()
    assert series.ww.semantic_tags == Categorical.standard_tags


def test_reset_semantic_tags_without_standard_tags(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    semantic_tags = 'initial_tag'
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)

    series.ww.reset_semantic_tags()
    assert series.ww.semantic_tags == set()


def test_remove_semantic_tags(sample_series):
    xfail_dask_and_koalas(sample_series)
    tags_to_remove = [
        'tag1',
        ['tag1'],
        {'tag1'}
    ]

    for tag in tags_to_remove:
        series = sample_series.astype('category')
        series.ww.init(semantic_tags=['tag1', 'tag2'], use_standard_tags=False)
        series.ww.remove_semantic_tags(tag)
        assert series.ww.semantic_tags == {'tag2'}


def test_remove_standard_semantic_tag(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    # Check that warning is raised if use_standard_tags is True - tag should be removed
    series.ww.init(logical_type=Categorical, semantic_tags='tag1', use_standard_tags=True)
    expected_message = "Removing standard semantic tag(s) 'category' from column 'sample_series'"
    with pytest.warns(UserWarning) as record:
        series.ww.remove_semantic_tags(['tag1', 'category'])
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message
    assert series.ww.semantic_tags == set()

    # Check that warning is not raised if use_standard_tags is False - tag should be removed
    series = sample_series.astype('category')
    series.ww.init(logical_type=Categorical, semantic_tags=['category', 'tag1'], use_standard_tags=False)

    with pytest.warns(None) as record:
        series.ww.remove_semantic_tags(['tag1', 'category'])
    assert len(record) == 0
    assert series.ww.semantic_tags == set()


def test_remove_semantic_tags_raises_error_with_invalid_tag(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init(semantic_tags='tag1')
    error_msg = re.escape("Semantic tag(s) 'invalid_tagname' not present on column 'sample_series'")
    with pytest.raises(LookupError, match=error_msg):
        series.ww.remove_semantic_tags('invalid_tagname')


def test_series_methods_on_accessor(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init()

    copied_series = series.ww.copy()
    assert copied_series is not series
    assert copied_series.ww._schema == series.ww._schema
    pd.testing.assert_series_equal(series, copied_series)


def test_series_methods_on_accessor_returning_series_valid_schema(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init()

    sorted_series = series.ww.sort_values()
    assert sorted_series.ww._schema == series.ww._schema
    assert sorted_series.ww._schema is not series.ww._schema
    pd.testing.assert_series_equal(sorted_series, series.sort_values())


def test_series_methods_on_accessor_inplace(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init()

    original_schema = series.ww._schema.copy()
    val = series.ww.pop(0)
    assert series.ww._schema == original_schema
    assert len(series) == 3
    assert val == 'a'


def test_series_methods_on_accessor_returning_series_invalid_schema(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init()

    warning = "Operation performed by astype has invalidated the Woodwork typing information:\n " \
        "dtype mismatch between original dtype, category, and returned dtype, string.\n " \
        "Please initialize Woodwork with Series.ww.init"

    with pytest.warns(TypingInfoMismatchWarning, match=warning):
        new_series = series.ww.astype('string')

    assert new_series.ww._schema is None


def test_series_methods_on_accessor_other_returns(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init()
    col_shape = series.ww.shape
    series_shape = series.shape
    assert col_shape == (4,)
    assert col_shape == series_shape

    assert series.name == series.ww.name
    assert series.nunique() == series.ww.nunique()


def test_series_methods_on_accessor_new_schema_dict(sample_series):
    xfail_dask_and_koalas(sample_series)

    series = sample_series.astype('category')
    series.ww.init(semantic_tags=['new_tag', 'tag2'], metadata={'important_keys': [1, 2, 3]})

    copied_series = series.ww.copy()

    assert copied_series.ww._schema == series.ww._schema
    assert copied_series.ww._schema is not series.ww._schema

    copied_series.ww.metadata['important_keys'].append(4)
    assert copied_series.ww.metadata['important_keys'] == [1, 2, 3, 4]
    assert series.ww.metadata['important_keys'] == [1, 2, 3]

    copied_series.ww.add_semantic_tags(['tag3'])
    assert copied_series.ww.semantic_tags == {'category', 'new_tag', 'tag2', 'tag3'}
    assert series.ww.semantic_tags == {'category', 'new_tag', 'tag2'}


def test_series_getattr_errors(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')

    error_message = "Woodwork not initialized for this Series. Initialize by calling Series.ww.init"
    with pytest.raises(AttributeError, match=error_message):
        series.ww.shape

    series.ww.init()
    error_message = "Woodwork has no attribute 'invalid_attr'"
    with pytest.raises(AttributeError, match=error_message):
        series.ww.invalid_attr


def test_ordinal_requires_instance_on_init(sample_series):
    xfail_dask_and_koalas(sample_series)
    error_msg = 'Must use an Ordinal instance with order values defined'
    with pytest.raises(TypeError, match=error_msg):
        sample_series.ww.init(logical_type=Ordinal)
    with pytest.raises(TypeError, match=error_msg):
        sample_series.ww.init(logical_type="Ordinal")


def test_ordinal_requires_instance_on_update(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init(logical_type="Categorical")

    error_msg = 'Must use an Ordinal instance with order values defined'
    with pytest.raises(TypeError, match=error_msg):
        series.ww.set_logical_type(Ordinal)
    with pytest.raises(TypeError, match=error_msg):
        series.ww.set_logical_type("Ordinal")


def test_ordinal_with_order(sample_series):
    if (ks and isinstance(sample_series, ks.Series)) or (dd and isinstance(sample_series, dd.Series)):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not compatible')

    series = sample_series.astype('category')
    ordinal_with_order = Ordinal(order=['a', 'b', 'c'])
    series.ww.init(logical_type=ordinal_with_order)
    assert isinstance(series.ww.logical_type, Ordinal)
    assert series.ww.logical_type.order == ['a', 'b', 'c']

    series = sample_series.astype('category')
    series.ww.init(logical_type='Categorical')
    new_series = series.ww.set_logical_type(ordinal_with_order)
    assert isinstance(new_series.ww.logical_type, Ordinal)
    assert new_series.ww.logical_type.order == ['a', 'b', 'c']


def test_ordinal_with_incomplete_ranking(sample_series):
    if (ks and isinstance(sample_series, ks.Series)) or (dd and isinstance(sample_series, dd.Series)):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not supported')

    series = sample_series.astype('category')
    ordinal_incomplete_order = Ordinal(order=['a', 'b'])
    error_msg = re.escape("Ordinal column sample_series contains values that are not "
                          "present in the order values provided: ['c']")
    with pytest.raises(ValueError, match=error_msg):
        series.ww.init(logical_type=ordinal_incomplete_order)


def test_ordinal_with_nan_values():
    nan_series = pd.Series(['a', 'b', np.nan, 'a']).astype('category')
    ordinal_with_order = Ordinal(order=['a', 'b'])
    nan_series.ww.init(logical_type=ordinal_with_order)
    assert isinstance(nan_series.ww.logical_type, Ordinal)
    assert nan_series.ww.logical_type.order == ['a', 'b']


# def test_latlong_formatting(latlongs):
#     expected_series = pd.Series([(1, 2), (3, 4)])
#     if ks and isinstance(latlongs[0], ks.Series):
#         expected_series = ks.Series([[1, 2], [3, 4]])
#     elif dd and isinstance(latlongs[0], dd.Series):
#         expected_series = dd.from_pandas(expected_series, npartitions=2)

#     expected_dc = DataColumn(expected_series, logical_type='LatLong', name='test_series')

#     for series in latlongs:
#         dc = DataColumn(series, logical_type='LatLong', name='test_series')
#         pd.testing.assert_series_equal(to_pandas(dc.to_series()), to_pandas(expected_series))

#         assert dc == expected_dc


def test_accessor_equality(sample_series, sample_datetime_series):
    xfail_dask_and_koalas(sample_series)
    xfail_dask_and_koalas(sample_datetime_series)

    # Check different parameters
    str_col = sample_series.astype('category')
    str_col.ww.init(logical_type='Categorical')
    str_col_2 = sample_series.astype('category')
    str_col_2.ww.init(logical_type=Categorical)
    str_col_diff_tags = sample_series.astype('category')
    str_col_diff_tags.ww.init(logical_type=Categorical, semantic_tags={'test'})
    diff_name_col = sample_datetime_series.astype('category')
    diff_name_col.ww.init(logical_type=Categorical)
    diff_dtype_col = sample_series.astype('string')
    diff_dtype_col.ww.init(logical_type=NaturalLanguage)
    diff_description_col = sample_series.astype('category')
    diff_description_col.ww.init(logical_type='Categorical', description='description')
    diff_metadata_col = sample_series.astype('category')
    diff_metadata_col.ww.init(logical_type='Categorical', metadata={'interesting_values': ['a', 'b']})

    assert str_col.ww == str_col_2.ww
    assert str_col.ww != str_col_diff_tags.ww
    assert str_col.ww != diff_name_col.ww
    assert str_col.ww != diff_dtype_col.ww
    assert str_col.ww != diff_description_col.ww
    assert str_col.ww != diff_metadata_col.ww

    # Check columns with same logical types but different parameters
    ordinal_ltype_1 = Ordinal(order=['a', 'b', 'c'])
    ordinal_ltype_2 = Ordinal(order=['b', 'a', 'c'])
    ordinal_col_1 = sample_series.astype('category')
    ordinal_col_2 = sample_series.astype('category')
    ordinal_col_1.ww.init(logical_type=ordinal_ltype_1)
    ordinal_col_2.ww.init(logical_type=ordinal_ltype_2)

    assert str_col.ww != ordinal_col_1.ww
    assert ordinal_col_1.ww != ordinal_col_2.ww
    assert ordinal_col_1.ww == ordinal_col_1.ww

    datetime_ltype_instantiated = Datetime(datetime_format='%Y-%m%d')
    datetime_col_format = sample_datetime_series.astype('datetime64[ns]')
    datetime_col_param = sample_datetime_series.astype('datetime64[ns]')
    datetime_col_instantiated = sample_datetime_series.astype('datetime64[ns]')
    datetime_col = sample_datetime_series.astype('datetime64[ns]')
    datetime_col_format.ww.init(logical_type=datetime_ltype_instantiated)
    datetime_col_param.ww.init(logical_type=Datetime(datetime_format=None))
    datetime_col_instantiated.ww.init(logical_type=Datetime())
    datetime_col.ww.init(logical_type=Datetime)

    assert datetime_col.ww != datetime_col_instantiated.ww
    assert datetime_col_instantiated.ww != datetime_col_format.ww
    assert datetime_col_instantiated.ww == datetime_col_param.ww

    # Check different underlying series
    str_col = sample_series.astype('string')
    str_col.ww.init(logical_type='NaturalLanguage')
    changed_series = sample_series.copy().replace(to_replace='a', value='test').astype('string')
    changed_series.ww.init(logical_type='NaturalLanguage')

    # We only check underlying data for equality with pandas dataframes
    if isinstance(str_col, pd.Series):
        assert str_col.ww != changed_series.ww
    else:
        assert str_col.ww == changed_series.ww


def test_accessor_metadata(sample_series):
    xfail_dask_and_koalas(sample_series)
    column_metadata = {'metadata_field': [1, 2, 3], 'created_by': 'user0'}

    series = sample_series.copy().astype('category')
    series.ww.init()
    assert series.ww.metadata == {}

    series = sample_series.copy().astype('category')
    series.ww.init(metadata=column_metadata)
    assert series.ww.metadata == column_metadata

    new_metadata = {'date_created': '1/1/19', 'created_by': 'user1'}

    series.ww.metadata = {**series.ww.metadata, **new_metadata}
    assert series.ww.metadata == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3], 'created_by': 'user1'}

    series.ww.metadata.pop('created_by')
    assert series.ww.metadata == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3]}

    series.ww.metadata['number'] = 1012034
    assert series.ww.metadata == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3], 'number': 1012034}


def test_accessor_metadata_error_on_init(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    err_msg = "Column metadata must be a dictionary"
    with pytest.raises(TypeError, match=err_msg):
        series.ww.init(metadata=123)


def test_accessor_metadata_error_on_update(sample_series):
    xfail_dask_and_koalas(sample_series)
    series = sample_series.astype('category')
    series.ww.init()
    err_msg = "Column metadata must be a dictionary"
    with pytest.raises(TypeError, match=err_msg):
        series.ww.metadata = 123
