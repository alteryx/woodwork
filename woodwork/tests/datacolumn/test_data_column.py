import re

import numpy as np
import pandas as pd
import pytest

from woodwork.datacolumn import DataColumn
from woodwork.exceptions import ColumnNameMismatchWarning, DuplicateTagsWarning
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
from woodwork.tests.testing_utils import to_pandas
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_datacolumn_init(sample_series):
    data_col = DataColumn(sample_series, use_standard_tags=False)
    # Koalas doesn't support category dtype
    if not (ks and isinstance(sample_series, ks.Series)):
        sample_series = sample_series.astype('category')
    pd.testing.assert_series_equal(to_pandas(data_col.to_series()), to_pandas(sample_series))
    assert data_col.name == sample_series.name
    assert data_col.logical_type == Categorical
    assert data_col.semantic_tags == set()


def test_datacolumn_init_with_logical_type(sample_series):
    data_col = DataColumn(sample_series, NaturalLanguage)
    assert data_col.logical_type == NaturalLanguage
    assert data_col.semantic_tags == set()

    data_col = DataColumn(sample_series, "natural_language")
    assert data_col.logical_type == NaturalLanguage
    assert data_col.semantic_tags == set()

    data_col = DataColumn(sample_series, "NaturalLanguage")
    assert data_col.logical_type == NaturalLanguage
    assert data_col.semantic_tags == set()


def test_datacolumn_init_with_semantic_tags(sample_series):
    semantic_tags = ['tag1', 'tag2']
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, use_standard_tags=False)
    assert data_col.semantic_tags == set(semantic_tags)


def test_datacolumn_init_wrong_series():
    error = 'Series must be one of: pandas.Series, dask.Series, koalas.Series, numpy.ndarray, or pandas.ExtensionArray'
    with pytest.raises(TypeError, match=error):
        DataColumn([1, 2, 3, 4])

    with pytest.raises(TypeError, match=error):
        DataColumn({1, 2, 3, 4})


def test_datacolumn_init_with_name(sample_series, sample_datetime_series):
    name = 'sample_series'
    changed_name = 'changed_name'

    dc_use_series_name = DataColumn(sample_series)
    assert dc_use_series_name.name == name
    assert dc_use_series_name.to_series().name == name

    warning = 'Name mismatch between sample_series and changed_name. DataColumn and underlying series name are now changed_name'
    with pytest.warns(ColumnNameMismatchWarning, match=warning):
        dc_use_input_name = DataColumn(sample_series, name=changed_name)
    assert dc_use_input_name.name == changed_name
    assert dc_use_input_name.to_series().name == changed_name

    warning = 'Name mismatch between sample_datetime_series and changed_name. DataColumn and underlying series name are now changed_name'
    with pytest.warns(ColumnNameMismatchWarning, match=warning):
        dc_with_ltype_change = DataColumn(sample_datetime_series, name=changed_name)
    assert dc_with_ltype_change.name == changed_name
    assert dc_with_ltype_change.to_series().name == changed_name


def test_datacolumn_inity_with_falsy_name(sample_series):
    falsy_name = 0
    warning = 'Name mismatch between sample_series and 0. DataColumn and underlying series name are now 0'
    with pytest.warns(ColumnNameMismatchWarning, match=warning):
        dc_falsy_name = DataColumn(sample_series.copy(), name=falsy_name)

    assert dc_falsy_name.name == falsy_name
    assert dc_falsy_name.to_series().name == falsy_name


def test_datacolumn_init_with_extension_array():
    series_categories = pd.Series([1, 2, 3], dtype='category')
    extension_categories = pd.Categorical([1, 2, 3])

    data_col = DataColumn(extension_categories)
    series = data_col.to_series()
    assert series.equals(series_categories)
    assert series.name is None
    assert data_col.name is None
    assert data_col.dtype == 'category'
    assert data_col.logical_type == Categorical

    series_ints = pd.Series([1, 2, None, 4], dtype='Int64')
    extension_ints = pd.arrays.IntegerArray(np.array([1, 2, 3, 4], dtype="int64"), mask=np.array([False, False, True, False]))

    data_col_with_name = DataColumn(extension_ints, name='extension')
    series = data_col_with_name.to_series()
    assert series.equals(series_ints)
    assert series.name == 'extension'
    assert data_col_with_name.name == 'extension'

    series_strs = pd.Series([1, 2, None, 4], dtype='string')

    data_col_different_ltype = DataColumn(extension_ints, logical_type='NaturalLanguage')
    series = data_col_different_ltype.to_series()
    assert series.equals(series_strs)
    assert data_col_different_ltype.logical_type == NaturalLanguage
    assert data_col_different_ltype.dtype == 'string'


def test_datacolumn_init_with_numpy_array():
    numpy_array = np.array([1, 2, 3, 4])
    expected_series = pd.Series([1, 2, 3, 4], dtype='Int64')

    dc = DataColumn(numpy_array)
    assert dc.name is None
    assert dc.logical_type == Integer
    assert dc.semantic_tags == {'numeric'}
    assert dc.dtype == 'Int64'
    assert dc._series.equals(expected_series)

    dc = DataColumn(numpy_array, logical_type='NaturalLanguage', name='test_col')
    expected_series.name = 'test_col'

    assert dc.name == 'test_col'
    assert dc.logical_type == NaturalLanguage
    assert dc.semantic_tags == set()
    assert dc.dtype == 'string'
    assert dc._series.equals(expected_series.astype('string'))


def test_datacolumn_with_alternate_semantic_tags_input(sample_series):
    semantic_tags = 'custom_tag'
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, use_standard_tags=False)
    assert data_col.semantic_tags == {'custom_tag'}

    semantic_tags = {'custom_tag', 'numeric'}
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, use_standard_tags=False)
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


def test_datacolumn_description(sample_series):
    column_description = "custom description"
    data_col = DataColumn(sample_series, description=column_description)
    assert data_col.description == column_description

    new_description = "updated description text"
    data_col.description = new_description
    assert data_col.description == new_description


def test_datacolumn_description_error(sample_series):
    err_msg = "Column description must be a string"
    with pytest.raises(TypeError, match=err_msg):
        DataColumn(sample_series, description=123)


def test_datacolumn_repr(sample_series):
    data_col = DataColumn(sample_series, use_standard_tags=False)
    # Koalas doesn't support categorical
    if ks and isinstance(sample_series, ks.Series):
        dtype = 'object'
    else:
        dtype = 'category'
    assert data_col.__repr__() == f'<DataColumn: sample_series (Physical Type = {dtype}) ' \
        '(Logical Type = Categorical) (Semantic Tags = set())>'


def test_set_semantic_tags(sample_series):
    semantic_tags = {'tag1', 'tag2'}
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, use_standard_tags=False)
    assert data_col.semantic_tags == semantic_tags

    new_tags = ['new_tag']
    new_col = data_col.set_semantic_tags(new_tags)
    assert new_col is not data_col
    assert new_col.semantic_tags == set(new_tags)


def test_set_semantic_tags_with_index(sample_series):
    semantic_tags = {'tag1', 'tag2'}
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, use_standard_tags=False)
    data_col._set_as_index()
    assert data_col.semantic_tags == {'tag1', 'tag2', 'index'}
    new_tags = ['new_tag']
    new_col = data_col.set_semantic_tags(new_tags)
    assert new_col.semantic_tags == {'index', 'new_tag'}
    new_col2 = new_col.set_semantic_tags(new_tags, retain_index_tags=False)
    assert new_col2.semantic_tags == {'new_tag'}


def test_set_semantic_tags_with_time_index(sample_datetime_series):
    semantic_tags = {'tag1', 'tag2'}
    data_col = DataColumn(sample_datetime_series, semantic_tags=semantic_tags, use_standard_tags=False)
    data_col._set_as_time_index()
    assert data_col.semantic_tags == {'tag1', 'tag2', 'time_index'}
    new_tags = ['new_tag']
    new_col = data_col.set_semantic_tags(new_tags)
    assert new_col.semantic_tags == {'time_index', 'new_tag'}
    new_col2 = new_col.set_semantic_tags(new_tags, retain_index_tags=False)
    assert new_col2.semantic_tags == {'new_tag'}


def test_adds_numeric_standard_tag():
    series = pd.Series([1, 2, 3])
    semantic_tags = 'custom_tag'

    logical_types = [Integer, Double]
    for logical_type in logical_types:
        data_col = DataColumn(series, logical_type=logical_type, semantic_tags=semantic_tags)
        assert data_col.semantic_tags == {'custom_tag', 'numeric'}


def test_adds_category_standard_tag():
    series = pd.Series([1, 2, 3])
    semantic_tags = 'custom_tag'

    logical_types = [Categorical, CountryCode, Ordinal(order=(1, 2, 3)), SubRegionCode, ZIPCode]
    for logical_type in logical_types:
        data_col = DataColumn(series, logical_type=logical_type, semantic_tags=semantic_tags)
        assert data_col.semantic_tags == {'custom_tag', 'category'}


def test_does_not_add_standard_tags():
    series = pd.Series([1, 2, 3])
    semantic_tags = 'custom_tag'
    data_col = DataColumn(series,
                          logical_type=Double,
                          semantic_tags=semantic_tags,
                          use_standard_tags=False)
    assert data_col.semantic_tags == {'custom_tag'}


def test_add_custom_tags(sample_series):
    semantic_tags = 'initial_tag'
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, use_standard_tags=False)

    new_col = data_col.add_semantic_tags('string_tag')
    assert new_col is not data_col
    assert new_col.semantic_tags == {'initial_tag', 'string_tag'}

    new_col2 = new_col.add_semantic_tags(['list_tag'])
    assert new_col2.semantic_tags == {'initial_tag', 'string_tag', 'list_tag'}

    new_col3 = new_col2.add_semantic_tags({'set_tag'})
    assert new_col3.semantic_tags == {'initial_tag', 'string_tag', 'list_tag', 'set_tag'}


def test_warns_on_setting_duplicate_tag(sample_series):
    semantic_tags = ['first_tag', 'second_tag']
    data_col = DataColumn(sample_series, semantic_tags=semantic_tags, use_standard_tags=False)

    expected_message = "Semantic tag(s) 'first_tag, second_tag' already present on column 'sample_series'"
    with pytest.warns(DuplicateTagsWarning) as record:
        data_col.add_semantic_tags(['first_tag', 'second_tag'])
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message


def test_set_logical_type_with_standard_tags(sample_series):
    data_col = DataColumn(sample_series,
                          logical_type=NaturalLanguage,
                          semantic_tags='original_tag',
                          use_standard_tags=True)

    new_col = data_col.set_logical_type(Categorical)
    assert isinstance(new_col, DataColumn)
    assert new_col is not data_col
    assert new_col.logical_type == Categorical
    assert new_col.semantic_tags == {'category'}


def test_set_logical_type_without_standard_tags(sample_series):
    data_col = DataColumn(sample_series,
                          logical_type=NaturalLanguage,
                          semantic_tags='original_tag',
                          use_standard_tags=False)

    new_col = data_col.set_logical_type(Categorical)
    assert isinstance(new_col, DataColumn)
    assert new_col is not data_col
    assert new_col.logical_type == Categorical
    assert new_col.semantic_tags == set()


def test_set_logical_type_retains_index_tag(sample_series):
    data_col = DataColumn(sample_series,
                          logical_type=NaturalLanguage,
                          semantic_tags='original_tag',
                          use_standard_tags=False)

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
                          use_standard_tags=False)

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
                          use_standard_tags=True)

    new_col = data_col.reset_semantic_tags()
    assert new_col is not data_col
    assert new_col.semantic_tags == Categorical.standard_tags


def test_reset_semantic_tags_without_standard_tags(sample_series):
    semantic_tags = 'initial_tag'
    data_col = DataColumn(sample_series,
                          semantic_tags=semantic_tags,
                          use_standard_tags=False)

    new_col = data_col.reset_semantic_tags()
    assert new_col is not data_col
    assert new_col.semantic_tags == set()


def test_reset_semantic_tags_with_index(sample_series):
    semantic_tags = 'initial_tag'
    data_col = DataColumn(sample_series,
                          semantic_tags=semantic_tags,
                          use_standard_tags=False)

    data_col._set_as_index()
    new_col = data_col.reset_semantic_tags(retain_index_tags=True)
    assert new_col.semantic_tags == {'index'}
    new_col = data_col.reset_semantic_tags()
    assert new_col.semantic_tags == set()


def test_reset_semantic_tags_with_time_index(sample_datetime_series):
    semantic_tags = 'initial_tag'
    data_col = DataColumn(sample_datetime_series,
                          semantic_tags=semantic_tags,
                          use_standard_tags=False)

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
                          use_standard_tags=False)

    for tag in tags_to_remove:
        new_col = data_col.remove_semantic_tags(tag)
        assert new_col is not data_col
        assert new_col.semantic_tags == {'tag2'}


def test_remove_standard_semantic_tag(sample_series):
    # Check that warning is raised if use_standard_tags is True - tag should be removed
    data_col = DataColumn(sample_series,
                          logical_type=Categorical,
                          semantic_tags='tag1',
                          use_standard_tags=True)
    expected_message = "Removing standard semantic tag(s) 'category' from column 'sample_series'"
    with pytest.warns(UserWarning) as record:
        new_col = data_col.remove_semantic_tags(['tag1', 'category'])
    assert len(record) == 1
    assert record[0].message.args[0] == expected_message
    assert new_col.semantic_tags == set()

    # Check that warning is not raised if use_standard_tags is False - tag should be removed
    data_col = DataColumn(sample_series,
                          logical_type=Categorical,
                          semantic_tags=['category', 'tag1'],
                          use_standard_tags=False)

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


def test_to_series(sample_series):
    data_col = DataColumn(sample_series)
    series = data_col.to_series()

    assert series is data_col._series
    pd.testing.assert_series_equal(to_pandas(series), to_pandas(data_col._series))


def test_shape(sample_series):
    col = DataColumn(sample_series)
    col_shape = col.shape
    series_shape = col.to_series().shape
    if dd and isinstance(sample_series, dd.Series):
        col_shape = (col_shape[0].compute(),)
        series_shape = (series_shape[0].compute(),)
    assert col_shape == (4,)
    assert col_shape == series_shape


def test_len(sample_series):
    col = DataColumn(sample_series)
    assert len(col) == len(sample_series) == 4


def test_dtype_update_on_init(sample_datetime_series):
    dc = DataColumn(sample_datetime_series,
                    logical_type='DateTime')
    assert dc._series.dtype == 'datetime64[ns]'


def test_dtype_update_on_ltype_change():
    dc = DataColumn(pd.Series([1, 2, 3]),
                    logical_type='Integer')
    assert dc._series.dtype == 'Int64'
    dc = dc.set_logical_type('Double')
    assert dc._series.dtype == 'float64'


def test_ordinal_requires_instance_on_init(sample_series):
    error_msg = 'Must use an Ordinal instance with order values defined'
    with pytest.raises(TypeError, match=error_msg):
        DataColumn(sample_series, logical_type=Ordinal)
    with pytest.raises(TypeError, match=error_msg):
        DataColumn(sample_series, logical_type="Ordinal")


def test_ordinal_requires_instance_on_update(sample_series):
    dc = DataColumn(sample_series, logical_type="NaturalLanguage")

    error_msg = 'Must use an Ordinal instance with order values defined'
    with pytest.raises(TypeError, match=error_msg):
        dc.set_logical_type(Ordinal)
    with pytest.raises(TypeError, match=error_msg):
        dc.set_logical_type("Ordinal")


def test_ordinal_with_order(sample_series):
    if (ks and isinstance(sample_series, ks.Series)) or (dd and isinstance(sample_series, dd.Series)):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not compatible')

    ordinal_with_order = Ordinal(order=['a', 'b', 'c'])
    dc = DataColumn(sample_series, logical_type=ordinal_with_order)
    assert isinstance(dc.logical_type, Ordinal)
    assert dc.logical_type.order == ['a', 'b', 'c']

    dc = DataColumn(sample_series, logical_type="NaturalLanguage")
    new_dc = dc.set_logical_type(ordinal_with_order)
    assert isinstance(new_dc.logical_type, Ordinal)
    assert new_dc.logical_type.order == ['a', 'b', 'c']


def test_ordinal_with_incomplete_ranking(sample_series):
    if (ks and isinstance(sample_series, ks.Series)) or (dd and isinstance(sample_series, dd.Series)):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not supported')

    ordinal_incomplete_order = Ordinal(order=['a', 'b'])
    error_msg = re.escape("Ordinal column sample_series contains values that are not "
                          "present in the order values provided: ['c']")
    with pytest.raises(ValueError, match=error_msg):
        DataColumn(sample_series, logical_type=ordinal_incomplete_order)


def test_ordinal_with_nan_values():
    nan_series = pd.Series(['a', 'b', np.nan, 'a'])
    ordinal_with_order = Ordinal(order=['a', 'b'])
    dc = DataColumn(nan_series, logical_type=ordinal_with_order)
    assert isinstance(dc.logical_type, Ordinal)
    assert dc.logical_type.order == ['a', 'b']


def test_latlong_formatting(latlongs):
    expected_series = pd.Series([(1, 2), (3, 4)])
    if ks and isinstance(latlongs[0], ks.Series):
        expected_series = ks.Series([[1, 2], [3, 4]])
    elif dd and isinstance(latlongs[0], dd.Series):
        expected_series = dd.from_pandas(expected_series, npartitions=2)

    expected_dc = DataColumn(expected_series, logical_type='LatLong', name='test_series')

    for series in latlongs:
        dc = DataColumn(series, logical_type='LatLong', name='test_series')
        pd.testing.assert_series_equal(to_pandas(dc.to_series()), to_pandas(expected_series))

        assert dc == expected_dc


def test_datacolumn_equality(sample_series, sample_datetime_series):
    # Check different parameters to DataColumn
    str_col = DataColumn(sample_series, logical_type='Categorical')
    str_col_2 = DataColumn(sample_series, logical_type=Categorical)
    str_col_diff_tags = DataColumn(sample_series, logical_type=Categorical, semantic_tags={'test'})
    diff_name_col = DataColumn(sample_datetime_series, logical_type=Categorical)
    diff_dtype_col = DataColumn(sample_series, logical_type=NaturalLanguage)
    diff_description_col = DataColumn(sample_series, logical_type='Categorical', description='description')
    diff_metadata_col = DataColumn(sample_series, logical_type='Categorical', metadata={'interesting_values': ['a', 'b']})

    assert str_col == str_col_2
    assert str_col != str_col_diff_tags
    assert str_col != diff_name_col
    assert str_col != diff_dtype_col
    assert str_col != diff_description_col
    assert str_col != diff_metadata_col

    # Check columns with same logical types but different parameters
    ordinal_ltype_1 = Ordinal(order=['a', 'b', 'c'])
    ordinal_ltype_2 = Ordinal(order=['b', 'a', 'c'])
    ordinal_col_1 = DataColumn(sample_series, logical_type=ordinal_ltype_1)
    ordinal_col_2 = DataColumn(sample_series, logical_type=ordinal_ltype_2)

    assert str_col != ordinal_col_1
    assert ordinal_col_1 != ordinal_col_2
    assert ordinal_col_1 == ordinal_col_1

    datetime_ltype_instantiated = Datetime(datetime_format='%Y-%m%d')
    datetime_col_format = DataColumn(sample_datetime_series, logical_type=datetime_ltype_instantiated)
    datetime_col_param = DataColumn(sample_datetime_series, logical_type=Datetime(datetime_format=None))
    datetime_col_instantiated = DataColumn(sample_datetime_series, logical_type=Datetime())
    datetime_col = DataColumn(sample_datetime_series, logical_type=Datetime)

    assert datetime_col != datetime_col_instantiated
    assert datetime_col_instantiated != datetime_col_format
    assert datetime_col_instantiated == datetime_col_param

    # Check different underlying series
    str_col = DataColumn(sample_series, logical_type='NaturalLanguage')
    changed_series = sample_series.copy().replace(to_replace='a', value='test')
    null_col = DataColumn(changed_series, logical_type='NaturalLanguage')

    # We only check underlying data for equality with pandas dataframes
    if isinstance(str_col.to_series(), pd.Series):
        assert str_col != null_col
    else:
        assert str_col == null_col


def test_datacolumn_metadata(sample_series):
    column_metadata = {'metadata_field': [1, 2, 3], 'created_by': 'user0'}

    data_col = DataColumn(sample_series)
    assert data_col.metadata == {}

    data_col = DataColumn(sample_series, metadata=column_metadata)
    assert data_col.metadata == column_metadata

    new_metadata = {'date_created': '1/1/19', 'created_by': 'user1'}

    data_col.metadata = {**data_col.metadata, **new_metadata}
    assert data_col.metadata == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3], 'created_by': 'user1'}

    data_col.metadata.pop('created_by')
    assert data_col.metadata == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3]}

    data_col.metadata['number'] = 1012034
    assert data_col.metadata == {'date_created': '1/1/19', 'metadata_field': [1, 2, 3], 'number': 1012034}


def test_datacolumn_metadata_error(sample_series):
    err_msg = "Column metadata must be a dictionary"
    with pytest.raises(TypeError, match=err_msg):
        DataColumn(sample_series, metadata=123)
