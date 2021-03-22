import re

import numpy as np
import pandas as pd
import pytest

from woodwork.accessor_utils import init_series
from woodwork.column_accessor import WoodworkColumnAccessor
from woodwork.exceptions import (
    DuplicateTagsWarning,
    StandardTagsChangedWarning,
    TypeConversionError,
    TypingInfoMismatchWarning
)
from woodwork.logical_types import (
    Categorical,
    CountryCode,
    Datetime,
    Double,
    Integer,
    LatLong,
    NaturalLanguage,
    Ordinal,
    SubRegionCode,
    ZIPCode
)
from woodwork.tests.testing_utils import (
    is_property,
    is_public_method,
    to_pandas
)
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_accessor_init(sample_series):
    assert sample_series.ww._schema is None
    sample_series.ww.init()
    assert isinstance(sample_series.ww._schema, dict)
    assert sample_series.ww.logical_type == Categorical
    assert sample_series.ww.semantic_tags == {'category'}


def test_accessor_init_with_logical_type(sample_series):
    series = sample_series.astype('string')
    series.ww.init(logical_type=NaturalLanguage)
    assert series.ww.logical_type == NaturalLanguage
    assert series.ww.semantic_tags == set()

    series = sample_series.astype('string')
    series.ww.init(logical_type="natural_language")
    assert series.ww.logical_type == NaturalLanguage
    assert series.ww.semantic_tags == set()

    series = sample_series.astype('string')
    series.ww.init(logical_type="NaturalLanguage")
    assert series.ww.logical_type == NaturalLanguage
    assert series.ww.semantic_tags == set()


def test_accessor_init_with_invalid_logical_type(sample_series):
    series = sample_series.astype('object')
    series_dtype = 'object'
    correct_dtype = 'string'
    error_message = f"Cannot initialize Woodwork. Series dtype '{series_dtype}' is incompatible with " \
        f"NaturalLanguage dtype. Try converting series dtype to '{correct_dtype}' before initializing " \
        "or use the woodwork.init_series function to initialize."
    with pytest.raises(ValueError, match=error_message):
        series.ww.init(logical_type=NaturalLanguage)


def test_accessor_init_with_semantic_tags(sample_series):
    semantic_tags = ['tag1', 'tag2']
    sample_series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)
    assert sample_series.ww.semantic_tags == set(semantic_tags)


def test_error_accessing_properties_before_init(sample_series):
    props_to_exclude = ['iloc', 'loc']
    props = [prop for prop in dir(sample_series.ww) if is_property(WoodworkColumnAccessor, prop) and prop not in props_to_exclude]

    error = "Woodwork not initialized for this Series. Initialize by calling Series.ww.init"
    for prop in props:
        with pytest.raises(AttributeError, match=error):
            getattr(sample_series.ww, prop)


def test_error_accessing_methods_before_init(sample_series):
    methods_to_exclude = ['init']
    public_methods = [method for method in dir(sample_series.ww) if is_public_method(WoodworkColumnAccessor, method)]
    public_methods = [method for method in public_methods if method not in methods_to_exclude]

    method_args_dict = {
        'add_semantic_tags': [{'new_tag'}],
        'remove_semantic_tags': [{'new_tag'}],
        'reset_semantic_tags': None,
        'set_logical_type': ['Integer'],
        'set_semantic_tags': [{'new_tag'}]
    }
    error = "Woodwork not initialized for this Series. Initialize by calling Series.ww.init"
    for method in public_methods:
        func = getattr(sample_series.ww, method)
        method_args = method_args_dict[method]
        with pytest.raises(AttributeError, match=error):
            if method_args:
                func(*method_args)
            else:
                func()


def test_accessor_with_alternate_semantic_tags_input(sample_series):
    series = sample_series.copy()
    semantic_tags = 'custom_tag'
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)
    assert series.ww.semantic_tags == {'custom_tag'}

    series = sample_series.copy()
    semantic_tags = {'custom_tag', 'numeric'}
    series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)
    assert series.ww.semantic_tags == semantic_tags


def test_logical_type_errors(sample_series):
    error_message = "Invalid logical type specified for 'sample_series'"
    with pytest.raises(TypeError, match=error_message):
        sample_series.ww.init(logical_type=int)

    error_message = "String naturalllanguage is not a valid logical type"
    with pytest.raises(ValueError, match=error_message):
        sample_series.ww.init(logical_type='naturalllanguage')


def test_semantic_tag_errors(sample_series):
    error_message = "semantic_tags for sample_series must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        sample_series.ww.init(semantic_tags=int)

    error_message = "semantic_tags for sample_series must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        sample_series.ww.init(semantic_tags={'index': {}, 'time_index': {}})

    error_message = "semantic_tags for sample_series must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        sample_series.ww.init(semantic_tags=['index', 1])


def test_accessor_description(sample_series):
    column_description = "custom description"
    sample_series.ww.init(description=column_description)
    assert sample_series.ww.description == column_description

    new_description = "updated description text"
    sample_series.ww.description = new_description
    assert sample_series.ww.description == new_description


def test_description_setter_error_before_init(sample_series):
    err_msg = "Woodwork not initialized for this Series. Initialize by calling Series.ww.init"
    with pytest.raises(AttributeError, match=err_msg):
        sample_series.ww.description = "description"


def test_description_error_on_init(sample_series):
    err_msg = "Column description must be a string"
    with pytest.raises(TypeError, match=err_msg):
        sample_series.ww.init(description=123)


def test_description_error_on_update(sample_series):
    sample_series.ww.init()
    err_msg = "Column description must be a string"
    with pytest.raises(TypeError, match=err_msg):
        sample_series.ww.description = 123


def test_accessor_repr(sample_series):
    sample_series.ww.init(use_standard_tags=False)
    # Koalas doesn't support categorical
    if ks and isinstance(sample_series, ks.Series):
        dtype = 'string'
    else:
        dtype = 'category'
    assert sample_series.ww.__repr__() == f'<Series: sample_series (Physical Type = {dtype}) ' \
        '(Logical Type = Categorical) (Semantic Tags = set())>'


def test_accessor_repr_error_before_init(sample_series):
    err_msg = "Woodwork not initialized for this Series. Initialize by calling Series.ww.init"
    with pytest.raises(AttributeError, match=err_msg):
        sample_series.ww.__repr__()


def test_set_semantic_tags(sample_series):
    semantic_tags = {'tag1', 'tag2'}
    sample_series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)
    assert sample_series.ww.semantic_tags == semantic_tags

    new_tags = ['new_tag']
    sample_series.ww.set_semantic_tags(new_tags)
    assert sample_series.ww.semantic_tags == set(new_tags)


def test_set_semantic_tags_with_standard_tags(sample_series):
    semantic_tags = {'tag1', 'tag2'}
    sample_series.ww.init(semantic_tags=semantic_tags, use_standard_tags=True)
    assert sample_series.ww.semantic_tags == semantic_tags.union({'category'})

    new_tags = ['new_tag']
    sample_series.ww.set_semantic_tags(new_tags)
    assert sample_series.ww.semantic_tags == set(new_tags).union({'category'})


def test_adds_numeric_standard_tag():
    series = pd.Series([1, 2, 3])
    semantic_tags = 'custom_tag'

    logical_types = [Integer, Double]
    for logical_type in logical_types:
        series = series.astype(logical_type.primary_dtype)
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
    semantic_tags = 'initial_tag'
    sample_series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)

    sample_series.ww.add_semantic_tags('string_tag')
    assert sample_series.ww.semantic_tags == {'initial_tag', 'string_tag'}

    sample_series.ww.add_semantic_tags(['list_tag'])
    assert sample_series.ww.semantic_tags == {'initial_tag', 'string_tag', 'list_tag'}

    sample_series.ww.add_semantic_tags({'set_tag'})
    assert sample_series.ww.semantic_tags == {'initial_tag', 'string_tag', 'list_tag', 'set_tag'}


def test_warns_on_adding_duplicate_tag(sample_series):
    semantic_tags = ['first_tag', 'second_tag']
    sample_series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)

    expected_message = "Semantic tag(s) 'first_tag, second_tag' already present on column 'sample_series'"
    with pytest.warns(DuplicateTagsWarning) as record:
        sample_series.ww.add_semantic_tags(['first_tag', 'second_tag'])
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message


def test_set_logical_type_with_standard_tags(sample_series):
    sample_series.ww.init(logical_type='Categorical',
                          semantic_tags='original_tag',
                          use_standard_tags=True)

    new_series = sample_series.ww.set_logical_type('CountryCode')
    assert sample_series.ww.logical_type == Categorical
    assert new_series.ww.logical_type == CountryCode
    assert new_series.ww.semantic_tags == {'category'}


def test_set_logical_type_without_standard_tags(sample_series):
    sample_series.ww.init(logical_type='Categorical',
                          semantic_tags='original_tag',
                          use_standard_tags=False)

    new_series = sample_series.ww.set_logical_type('CountryCode')
    assert sample_series.ww.logical_type == Categorical
    assert new_series.ww.logical_type == CountryCode
    assert new_series.ww.semantic_tags == set()


def test_set_logical_type_valid_dtype_change(sample_series):
    sample_series.ww.init(logical_type='Categorical')

    new_series = sample_series.ww.set_logical_type('NaturalLanguage')

    if ks and isinstance(sample_series, ks.Series):
        # Koalas uses string dtype for Categorical
        original_dtype = 'string'
    else:
        original_dtype = 'category'
    new_dtype = 'string'

    assert sample_series.ww.logical_type == Categorical
    assert sample_series.dtype == original_dtype
    assert new_series.ww.logical_type == NaturalLanguage
    assert new_series.dtype == new_dtype


def test_set_logical_type_invalid_dtype_change(sample_series):
    if dd and isinstance(sample_series, dd.Series):
        pytest.xfail('Dask type conversion with astype does not fail until compute is called')
    if ks and isinstance(sample_series, ks.Series):
        pytest.xfail('Koalas allows this conversion, filling values it cannot convert with NaN '
                     'and converting dtype to float.')
    sample_series.ww.init(logical_type='Categorical')
    error_message = "Error converting datatype for sample_series from type category to " \
        "type Int64. Please confirm the underlying data is consistent with logical type Integer."
    with pytest.raises(TypeConversionError, match=error_message):
        sample_series.ww.set_logical_type('Integer')


def test_reset_semantic_tags_with_standard_tags(sample_series):
    semantic_tags = 'initial_tag'
    sample_series.ww.init(semantic_tags=semantic_tags,
                          logical_type=Categorical,
                          use_standard_tags=True)

    sample_series.ww.reset_semantic_tags()
    assert sample_series.ww.semantic_tags == Categorical.standard_tags


def test_reset_semantic_tags_without_standard_tags(sample_series):
    semantic_tags = 'initial_tag'
    sample_series.ww.init(semantic_tags=semantic_tags, use_standard_tags=False)

    sample_series.ww.reset_semantic_tags()
    assert sample_series.ww.semantic_tags == set()


def test_remove_semantic_tags(sample_series):
    tags_to_remove = [
        'tag1',
        ['tag1'],
        {'tag1'}
    ]

    for tag in tags_to_remove:
        series = sample_series.copy()
        series.ww.init(semantic_tags=['tag1', 'tag2'], use_standard_tags=False)
        series.ww.remove_semantic_tags(tag)
        assert series.ww.semantic_tags == {'tag2'}


def test_remove_standard_semantic_tag(sample_series):
    series = sample_series.copy()
    # Check that warning is raised if use_standard_tags is True - tag should be removed
    series.ww.init(logical_type=Categorical, semantic_tags='tag1', use_standard_tags=True)
    expected_message = 'Standard tags have been removed from "sample_series"'
    with pytest.warns(StandardTagsChangedWarning) as record:
        series.ww.remove_semantic_tags(['tag1', 'category'])
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message
    assert series.ww.semantic_tags == set()

    # Check that warning is not raised if use_standard_tags is False - tag should be removed
    series = sample_series.copy()
    series.ww.init(logical_type=Categorical, semantic_tags=['category', 'tag1'], use_standard_tags=False)

    with pytest.warns(None) as record:
        series.ww.remove_semantic_tags(['tag1', 'category'])
    assert len(record) == 0
    assert series.ww.semantic_tags == set()


def test_remove_semantic_tags_raises_error_with_invalid_tag(sample_series):
    sample_series.ww.init(semantic_tags='tag1')
    error_msg = re.escape("Semantic tag(s) 'invalid_tagname' not present on column 'sample_series'")
    with pytest.raises(LookupError, match=error_msg):
        sample_series.ww.remove_semantic_tags('invalid_tagname')


def test_series_methods_on_accessor(sample_series):
    sample_series.ww.init()

    copied_series = sample_series.ww.copy()
    assert copied_series is not sample_series
    assert copied_series.ww._schema == sample_series.ww._schema
    pd.testing.assert_series_equal(to_pandas(sample_series), to_pandas(copied_series))


def test_series_methods_on_accessor_without_standard_tags(sample_series):
    sample_series.ww.init(use_standard_tags=False)

    copied_series = sample_series.ww.copy()
    assert copied_series is not sample_series
    assert copied_series.ww._schema == sample_series.ww._schema
    pd.testing.assert_series_equal(to_pandas(sample_series), to_pandas(copied_series))


def test_series_methods_on_accessor_returning_series_valid_schema(sample_series):
    if ks and isinstance(sample_series, ks.Series):
        pytest.xfail('Running replace on Koalas series changes series dtype to object, invalidating schema')
    sample_series.ww.init()

    replace_series = sample_series.ww.replace('a', 'd')
    assert replace_series.ww._schema == sample_series.ww._schema
    assert replace_series.ww._schema is not sample_series.ww._schema
    pd.testing.assert_series_equal(to_pandas(replace_series), to_pandas(sample_series.replace('a', 'd')))


def test_series_methods_on_accessor_dtype_mismatch(sample_df):
    ints_series = sample_df['id'].astype('Int64')
    ints_series.ww.init()

    assert ints_series.ww.logical_type == Integer
    assert str(ints_series.dtype) == 'Int64'

    warning = ("Operation performed by astype has invalidated the Woodwork typing information:\n "
               "dtype mismatch between original dtype, Int64, and returned dtype, int64.\n "
               "Please initialize Woodwork with Series.ww.init")
    with pytest.warns(TypingInfoMismatchWarning, match=warning):
        series = ints_series.ww.astype('int64')
    assert series.ww._schema is None


def test_series_methods_on_accessor_inplace(sample_series):
    # TODO: Try to find a supported inplace method for Dask, if one exists
    if dd and isinstance(sample_series, dd.Series):
        pytest.xfail('Dask does not support pop.')
    sample_series.ww.init()

    original_schema = sample_series.ww._schema.copy()
    val = sample_series.ww.pop(0)
    assert sample_series.ww._schema == original_schema
    assert len(sample_series) == 3
    assert val == 'a'


def test_series_methods_on_accessor_returning_series_invalid_schema(sample_series):
    sample_series.ww.init()

    if ks and isinstance(sample_series, ks.Series):
        # Koalas uses `string` for Categorical, so must try a different conversion
        original_type = 'string'
        new_type = 'Int64'
    else:
        original_type = 'category'
        new_type = 'string'

    warning = "Operation performed by astype has invalidated the Woodwork typing information:\n " \
        f"dtype mismatch between original dtype, {original_type}, and returned dtype, {new_type}.\n " \
        "Please initialize Woodwork with Series.ww.init"

    with pytest.warns(TypingInfoMismatchWarning, match=warning):
        new_series = sample_series.ww.astype(new_type)

    assert new_series.ww._schema is None


def test_series_methods_on_accessor_other_returns(sample_series):
    sample_series.ww.init()
    col_shape = sample_series.ww.shape
    series_shape = sample_series.shape
    if dd and isinstance(sample_series, dd.Series):
        col_shape = (col_shape[0].compute(),)
        series_shape = (series_shape[0].compute())
    assert col_shape == (4,)
    assert col_shape == series_shape

    assert sample_series.name == sample_series.ww.name
    series_nunique = sample_series.nunique()
    ww_nunique = sample_series.ww.nunique()
    if dd and isinstance(sample_series, dd.Series):
        series_nunique = series_nunique.compute()
        ww_nunique = ww_nunique.compute()
    assert series_nunique == ww_nunique


def test_series_methods_on_accessor_new_schema_dict(sample_series):
    sample_series.ww.init(semantic_tags=['new_tag', 'tag2'], metadata={'important_keys': [1, 2, 3]})

    copied_series = sample_series.ww.copy()

    assert copied_series.ww._schema == sample_series.ww._schema
    assert copied_series.ww._schema is not sample_series.ww._schema

    copied_series.ww.metadata['important_keys'].append(4)
    assert copied_series.ww.metadata['important_keys'] == [1, 2, 3, 4]
    assert sample_series.ww.metadata['important_keys'] == [1, 2, 3]

    copied_series.ww.add_semantic_tags(['tag3'])
    assert copied_series.ww.semantic_tags == {'category', 'new_tag', 'tag2', 'tag3'}
    assert sample_series.ww.semantic_tags == {'category', 'new_tag', 'tag2'}


def test_series_getattr_errors(sample_series):
    error_message = "Woodwork not initialized for this Series. Initialize by calling Series.ww.init"
    with pytest.raises(AttributeError, match=error_message):
        sample_series.ww.shape

    sample_series.ww.init()
    error_message = "Woodwork has no attribute 'invalid_attr'"
    with pytest.raises(AttributeError, match=error_message):
        sample_series.ww.invalid_attr


def test_ordinal_requires_instance_on_init(sample_series):
    error_msg = 'Must use an Ordinal instance with order values defined'
    with pytest.raises(TypeError, match=error_msg):
        sample_series.ww.init(logical_type=Ordinal)
    with pytest.raises(TypeError, match=error_msg):
        sample_series.ww.init(logical_type="Ordinal")


def test_ordinal_requires_instance_on_update(sample_series):
    sample_series.ww.init(logical_type="Categorical")

    error_msg = 'Must use an Ordinal instance with order values defined'
    with pytest.raises(TypeError, match=error_msg):
        sample_series.ww.set_logical_type(Ordinal)
    with pytest.raises(TypeError, match=error_msg):
        sample_series.ww.set_logical_type("Ordinal")


def test_ordinal_with_order(sample_series):
    if (ks and isinstance(sample_series, ks.Series)) or (dd and isinstance(sample_series, dd.Series)):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not compatible')

    series = sample_series.copy()
    ordinal_with_order = Ordinal(order=['a', 'b', 'c'])
    series.ww.init(logical_type=ordinal_with_order)
    assert isinstance(series.ww.logical_type, Ordinal)
    assert series.ww.logical_type.order == ['a', 'b', 'c']

    series = sample_series.copy()
    series.ww.init(logical_type='Categorical')
    new_series = series.ww.set_logical_type(ordinal_with_order)
    assert isinstance(new_series.ww.logical_type, Ordinal)
    assert new_series.ww.logical_type.order == ['a', 'b', 'c']


def test_ordinal_with_incomplete_ranking(sample_series):
    if (ks and isinstance(sample_series, ks.Series)) or (dd and isinstance(sample_series, dd.Series)):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not supported')

    ordinal_incomplete_order = Ordinal(order=['a', 'b'])
    error_msg = re.escape("Ordinal column sample_series contains values that are not "
                          "present in the order values provided: ['c']")
    with pytest.raises(ValueError, match=error_msg):
        sample_series.ww.init(logical_type=ordinal_incomplete_order)


def test_ordinal_with_nan_values():
    nan_series = pd.Series(['a', 'b', np.nan, 'a']).astype('category')
    ordinal_with_order = Ordinal(order=['a', 'b'])
    nan_series.ww.init(logical_type=ordinal_with_order)
    assert isinstance(nan_series.ww.logical_type, Ordinal)
    assert nan_series.ww.logical_type.order == ['a', 'b']


def test_latlong_init_with_valid_series(latlongs):
    series = latlongs[0]
    series.ww.init(logical_type="LatLong")
    assert series.ww.logical_type == LatLong


def test_latlong_init_error_with_invalid_series(latlongs):
    series = latlongs[1]
    error_message = "Cannot initialize Woodwork. Series does not contain properly formatted " \
        "LatLong data. Try reformatting before initializing or use the " \
        "woodwork.init_series function to initialize."
    with pytest.raises(ValueError, match=error_message):
        series.ww.init(logical_type="LatLong")


def test_latlong_formatting_with_init_series(latlongs):
    expected_series = pd.Series([(1.0, 2.0), (3.0, 4.0)])
    if dd and isinstance(latlongs[0], dd.Series):
        expected_series = dd.from_pandas(expected_series, npartitions=2)
    elif ks and isinstance(latlongs[0], ks.Series):
        expected_series = ks.Series([[1.0, 2.0], [3.0, 4.0]])

    expected_series.ww.init(logical_type=LatLong)
    for series in latlongs:
        new_series = init_series(series, logical_type=LatLong)
        assert new_series.ww.logical_type == LatLong
        pd.testing.assert_series_equal(to_pandas(new_series), to_pandas(expected_series))
        assert expected_series.ww._schema == new_series.ww._schema


def test_accessor_equality(sample_series, sample_datetime_series):
    # Check different parameters
    str_col = sample_series.copy()
    str_col.ww.init(logical_type='Categorical')
    str_col_2 = sample_series.copy()
    str_col_2.ww.init(logical_type=Categorical)
    str_col_diff_tags = sample_series.copy()
    str_col_diff_tags.ww.init(logical_type=Categorical, semantic_tags={'test'})
    if ks and isinstance(sample_datetime_series, ks.Series):
        diff_name_col = sample_datetime_series.astype('string')
    else:
        diff_name_col = sample_datetime_series.astype('category')
    diff_name_col.ww.init(logical_type=Categorical)
    diff_dtype_col = sample_series.astype('string')
    diff_dtype_col.ww.init(logical_type=NaturalLanguage)
    diff_description_col = sample_series.copy()
    diff_description_col.ww.init(logical_type='Categorical', description='description')
    diff_metadata_col = sample_series.copy()
    diff_metadata_col.ww.init(logical_type='Categorical', metadata={'interesting_values': ['a', 'b']})

    assert str_col.ww == str_col_2.ww
    assert str_col.ww != str_col_diff_tags.ww
    if isinstance(str_col, pd.Series) and isinstance(diff_name_col, pd.Series):
        # Name is from series, series equality checked only with Pandas input
        assert str_col.ww != diff_name_col.ww
    assert str_col.ww != diff_dtype_col.ww
    assert str_col.ww != diff_description_col.ww
    assert str_col.ww != diff_metadata_col.ww

    # Check columns with same logical types but different parameters
    ordinal_ltype_1 = Ordinal(order=['a', 'b', 'c'])
    ordinal_ltype_2 = Ordinal(order=['b', 'a', 'c'])
    ordinal_col_1 = sample_series.copy()
    ordinal_col_2 = sample_series.copy()
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
    changed_series = sample_series.copy().replace(to_replace='a', value='test')
    changed_series = changed_series.astype('string')
    changed_series.ww.init(logical_type='NaturalLanguage')

    # We only check underlying data for equality with pandas dataframes
    if isinstance(str_col, pd.Series):
        assert str_col.ww != changed_series.ww
    else:
        assert str_col.ww == changed_series.ww


def test_accessor_metadata(sample_series):
    column_metadata = {'metadata_field': [1, 2, 3], 'created_by': 'user0'}

    series = sample_series.copy()
    series.ww.init()
    assert series.ww.metadata == {}

    series = sample_series.copy()
    series.ww.init(metadata=column_metadata)
    assert series.ww.metadata == column_metadata

    new_metadata = {'date_created': '1/1/19', 'created_by': 'user1'}

    series.ww.metadata = {**series.ww.metadata, **new_metadata}
    assert series.ww.metadata == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3], 'created_by': 'user1'}

    series.ww.metadata.pop('created_by')
    assert series.ww.metadata == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3]}

    series.ww.metadata['number'] = 1012034
    assert series.ww.metadata == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3], 'number': 1012034}


def test_metadata_setter_error_before_init(sample_series):
    err_msg = "Woodwork not initialized for this Series. Initialize by calling Series.ww.init"
    with pytest.raises(AttributeError, match=err_msg):
        sample_series.ww.metadata = {"key": "val"}


def test_accessor_metadata_error_on_init(sample_series):
    err_msg = "Column metadata must be a dictionary"
    with pytest.raises(TypeError, match=err_msg):
        sample_series.ww.init(metadata=123)


def test_accessor_metadata_error_on_update(sample_series):
    sample_series.ww.init()
    err_msg = "Column metadata must be a dictionary"
    with pytest.raises(TypeError, match=err_msg):
        sample_series.ww.metadata = 123


def test_non_string_column_name(sample_series):
    sample_series.name = 0
    sample_series.ww.init(semantic_tags={'test_tag'})

    assert sample_series.ww.name == 0
    assert sample_series.name == 0
    assert sample_series.ww.semantic_tags == {'category', 'test_tag'}
