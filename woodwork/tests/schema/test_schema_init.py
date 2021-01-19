import re

import numpy as np
import pandas as pd
import pytest

import woodwork as ww
from woodwork import Schema
from woodwork.logical_types import (
    URL,
    Boolean,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    EmailAddress,
    Filepath,
    FullName,
    Integer,
    IPAddress,
    LatLong,
    LogicalType,
    NaturalLanguage,
    Ordinal,
    PhoneNumber,
    SubRegionCode,
    Timedelta,
    ZIPCode
)
from woodwork.schema import (
    _check_column_metadata,
    _check_index,
    _check_logical_types,
    _check_semantic_tags,
    _check_table_metadata,
    _check_time_index,
    _check_unique_column_names,
    _validate_dataframe,
    _validate_params
)
from woodwork.tests.testing_utils import (
    check_column_order,
    mi_between_cols,
    to_pandas,
    validate_subset_dt
)
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
dask_delayed = import_or_none('dask.delayed')
ks = import_or_none('databricks.koalas')


def test_schema_init(sample_df):
    schema = Schema(sample_df)

    assert schema.name is None
    assert schema.index is None
    assert schema.time_index is None

    assert set(schema.columns.keys()) == set(sample_df.columns)


def test_schema_init_with_name(sample_df):
    schema = Schema(sample_df,
                    name='schema')

    assert schema.name == 'schema'
    assert schema.index == None
    assert schema.time_index == None


def test_schema_init_with_name_and_index(sample_df):
    schema = Schema(sample_df,
                    name='schema',
                    index='id',
                    time_index='signup_date')

    assert schema.name == 'schema'
    assert schema.index == 'id'
    assert schema.time_index == 'signup_date'
    assert schema.columns[schema.time_index]['logical_type'] == Datetime


def test_schema_init_with_valid_string_time_index(time_index_df):
    dt = Schema(time_index_df,
                name='schema',
                index='id',
                time_index='times')

    assert dt.name == 'schema'
    assert dt.index == 'id'
    assert dt.time_index == 'times'
    assert dt.columns[dt.time_index]['logical_type'] == Datetime


def test_schema_init_with_invalid_string_time_index(sample_df):
    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        Schema(sample_df, name='schema', time_index='full_name')


def test_schema_init_with_logical_types(sample_df):
    logical_types = {
        'full_name': NaturalLanguage,
        'age': Double
    }
    dt = Schema(sample_df,
                name='schema',
                logical_types=logical_types)
    assert dt.columns['full_name']['logical_type'] == NaturalLanguage
    assert dt.columns['age']['logical_type'] == Double


def test_schema_init_with_string_logical_types(sample_df):
    logical_types = {
        'full_name': 'natural_language',
        'age': 'Double'
    }
    dt = Schema(sample_df,
                name='schema',
                logical_types=logical_types)
    assert dt.columns['full_name']['logical_type'] == NaturalLanguage
    assert dt.columns['age']['logical_type'] == Double

    logical_types = {
        'full_name': 'NaturalLanguage',
        'age': 'Integer',
        'signup_date': 'Datetime'
    }
    dt = Schema(sample_df,
                name='schema',
                logical_types=logical_types
                # --> add time index back in after implementing
                # ,
                # time_index='signup_date'
                )
    assert dt.columns['full_name']['logical_type'] == NaturalLanguage
    assert dt.columns['age']['logical_type'] == Integer
    # assert dt.time_index == 'signup_date'


def test_schema_init_with_semantic_tags(sample_df):
    semantic_tags = {
        'id': 'custom_tag',
    }
    dt = Schema(sample_df,
                name='schema',
                semantic_tags=semantic_tags,
                use_standard_tags=False)

    id_semantic_tags = dt.columns['id']['semantic_tags']
    assert isinstance(id_semantic_tags, set)
    assert len(id_semantic_tags) == 1
    assert 'custom_tag' in id_semantic_tags


def test_schema_adds_standard_semantic_tags(sample_df):
    dt = Schema(sample_df,
                name='schema',
                logical_types={
                    'id': Categorical,
                    'age': Integer,
                })

    assert dt.semantic_tags['id'] == {'category'}
    assert dt.semantic_tags['age'] == {'numeric'}


def test_semantic_tags_during_init(sample_df):
    semantic_tags = {
        'full_name': 'tag1',
        'email': ['tag2'],
        'phone_number': ['tag3'],
        'signup_date': ['secondary_time_index'],
        'age': ['numeric', 'age']
    }
    expected_types = {
        'full_name': {'tag1'},
        'email': {'tag2'},
        'phone_number': {'tag3'},
        'signup_date': {'secondary_time_index'},
        'age': {'numeric', 'age'}
    }
    dt = Schema(sample_df, semantic_tags=semantic_tags)
    assert dt.columns['full_name']['semantic_tags'] == expected_types['full_name']
    assert dt.columns['email']['semantic_tags'] == expected_types['email']
    assert dt.columns['phone_number']['semantic_tags'] == expected_types['phone_number']
    assert dt.columns['signup_date']['semantic_tags'] == expected_types['signup_date']
    assert dt.columns['age']['semantic_tags'] == expected_types['age']


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


def test_validate_params_errors(sample_df):
    error_message = 'Dataframe must be one of: pandas.DataFrame, dask.DataFrame, koalas.DataFrame, numpy.ndarray'
    with pytest.raises(TypeError, match=error_message):
        _validate_dataframe(dataframe=pd.Series())

    error_message = 'DataTable name must be a string'
    with pytest.raises(TypeError, match=error_message):
        _validate_params(dataframe=sample_df,
                         name=1,
                         index=None,
                         time_index=None,
                         logical_types=None,
                         table_metadata=None,
                         column_metadata=None,
                         semantic_tags=None,
                         make_index=False,
                         column_descriptions=None)


def test_check_index_errors(sample_df):
    error_message = 'Specified index column `foo` not found in dataframe. To create a new index column, set make_index to True.'
    with pytest.raises(LookupError, match=error_message):
        _check_index(dataframe=sample_df, index='foo')

    if isinstance(sample_df, pd.DataFrame):
        # Does not check for index uniqueness with Dask
        error_message = 'Index column must be unique'
        with pytest.raises(LookupError, match=error_message):
            _check_index(sample_df, index='age')

    error_message = 'When setting make_index to True, the name specified for index cannot match an existing column name'
    with pytest.raises(IndexError, match=error_message):
        _check_index(sample_df, index='id', make_index=True)

    error_message = 'When setting make_index to True, the name for the new index must be specified in the index parameter'
    with pytest.raises(IndexError, match=error_message):
        _check_index(sample_df, index=None, make_index=True)


def test_check_time_index_errors(sample_df):
    error_message = 'Specified time index column `foo` not found in dataframe'
    with pytest.raises(LookupError, match=error_message):
        _check_time_index(dataframe=sample_df, time_index='foo')


def test_check_unique_column_names(sample_df):
    if ks and isinstance(sample_df, ks.DataFrame):
        pytest.skip("Koalas enforces unique column names")
    duplicate_cols_df = sample_df.copy()
    if dd and isinstance(sample_df, dd.DataFrame):
        duplicate_cols_df = dd.concat([duplicate_cols_df, duplicate_cols_df['age']], axis=1)
    else:
        duplicate_cols_df.insert(0, 'age', [18, 21, 65, 43], allow_duplicates=True)
    with pytest.raises(IndexError, match='Dataframe cannot contain duplicate columns names'):
        _check_unique_column_names(duplicate_cols_df)


def test_check_logical_types_errors(sample_df):
    error_message = 'logical_types must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_logical_types(sample_df, logical_types='type')

    bad_logical_types_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("logical_types contains columns that are not present in dataframe: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_logical_types(sample_df, bad_logical_types_keys)


def test_check_semantic_tags_errors(sample_df):
    error_message = 'semantic_tags must be a dictionary'
    with pytest.raises(TypeError, match=error_message):
        _check_semantic_tags(sample_df, semantic_tags='type')

    bad_semantic_tags_keys = {
        'full_name': None,
        'age': None,
        'birthday': None,
        'occupation': None,
    }
    error_message = re.escape("semantic_tags contains columns that are not present in dataframe: ['birthday', 'occupation']")
    with pytest.raises(LookupError, match=error_message):
        _check_semantic_tags(sample_df, bad_semantic_tags_keys)


def test_check_table_metadata_errors():
    error_message = 'Table metadata must be a dictionary.'
    with pytest.raises(TypeError, match=error_message):
        _check_table_metadata('test')


def test_check_column_metadata_errors(sample_df):
    error_message = 'Column metadata must be a dictionary.'
    with pytest.raises(TypeError, match=error_message):
        _check_column_metadata(sample_df, column_metadata='test')

    column_metadata = {
        'invalid_col': {'description': 'not a valid column'}
    }
    err_msg = re.escape("column_metadata contains columns that are not present in dataframe: ['invalid_col']")
    with pytest.raises(LookupError, match=err_msg):
        Schema(sample_df, column_metadata=column_metadata)


def test_raises_error_setting_index_tag_directly(sample_df):
    error_msg = re.escape("Cannot add 'index' tag directly. To set a column as the index, "
                          "use DataTable.set_index() instead.")
    with pytest.raises(ValueError, match=error_msg):
        Schema(sample_df, semantic_tags={'id': 'index'})

    # --> add back in when allowing updating tags
    # Schema = Schema(sample_df)
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.add_semantic_tags({'id': 'index'})
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.set_semantic_tags({'id': 'index'})


def test_raises_error_setting_time_index_tag_directly(sample_df):
    error_msg = re.escape("Cannot add 'time_index' tag directly. To set a column as the time index, "
                          "use DataTable.set_time_index() instead.")
    with pytest.raises(ValueError, match=error_msg):
        Schema(sample_df, semantic_tags={'signup_date': 'time_index'})

    # --> add back in when allowing updating tags
    # schema = Schema(sample_series)
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.add_semantic_tags({'signup_date': 'time_index'})
    # with pytest.raises(ValueError, match=error_msg):
    #     schema.set_semantic_tags({'signup_date': 'time_index'})


def test_int_dtype_inference_on_init():
    df = pd.DataFrame({
        'ints_no_nans': pd.Series([1, 2]),
        'ints_nan': pd.Series([1, np.nan]),
        'ints_NA': pd.Series([1, pd.NA]),
        'ints_NA_specified': pd.Series([1, pd.NA], dtype='Int64')})
    Schema(df)

    assert df['ints_no_nans'].dtype == 'Int64'
    assert df['ints_nan'].dtype == 'float64'
    assert df['ints_NA'].dtype == 'category'
    assert df['ints_NA_specified'].dtype == 'Int64'


def test_bool_dtype_inference_on_init():
    df = pd.DataFrame({
        'bools_no_nans': pd.Series([True, False]),
        'bool_nan': pd.Series([True, np.nan]),
        'bool_NA': pd.Series([True, pd.NA]),
        'bool_NA_specified': pd.Series([True, pd.NA], dtype="boolean")})
    Schema(df)

    assert df['bools_no_nans'].dtype == 'boolean'
    assert df['bool_nan'].dtype == 'category'
    assert df['bool_NA'].dtype == 'category'
    assert df['bool_NA_specified'].dtype == 'boolean'


def test_str_dtype_inference_on_init():
    df = pd.DataFrame({
        'str_no_nans': pd.Series(['a', 'b']),
        'str_nan': pd.Series(['a', np.nan]),
        'str_NA': pd.Series(['a', pd.NA]),
        'str_NA_specified': pd.Series([1, pd.NA], dtype="string"),
        'long_str_NA_specified': pd.Series(['this is a very long sentence inferred as a string', pd.NA], dtype="string"),
        'long_str_NA': pd.Series(['this is a very long sentence inferred as a string', pd.NA])
    })
    Schema(df)

    assert df['str_no_nans'].dtype == 'category'
    assert df['str_nan'].dtype == 'category'
    assert df['str_NA'].dtype == 'category'
    assert df['str_NA_specified'].dtype == 'category'
    assert df['long_str_NA_specified'].dtype == 'string'
    assert df['long_str_NA'].dtype == 'string'


def test_float_dtype_inference_on_init():
    df = pd.DataFrame({
        'floats_no_nans': pd.Series([1.1, 2.2]),
        'floats_nan': pd.Series([1.1, np.nan]),
        'floats_NA': pd.Series([1.1, pd.NA]),
        'floats_nan_specified': pd.Series([1.1, np.nan], dtype='float')})
    Schema(df)

    assert df['floats_no_nans'].dtype == 'float64'
    assert df['floats_nan'].dtype == 'float64'
    assert df['floats_NA'].dtype == 'category'
    assert df['floats_nan_specified'].dtype == 'float64'


def test_datetime_dtype_inference_on_init():
    df = pd.DataFrame({
        'date_no_nans': pd.Series([pd.to_datetime('2020-09-01')] * 2),
        'date_nan': pd.Series([pd.to_datetime('2020-09-01'), np.nan]),
        'date_NA': pd.Series([pd.to_datetime('2020-09-01'), pd.NA]),
        'date_NaT': pd.Series([pd.to_datetime('2020-09-01'), pd.NaT]),
        'date_NA_specified': pd.Series([pd.to_datetime('2020-09-01'), pd.NA], dtype='datetime64[ns]')})
    Schema(df)

    assert df['date_no_nans'].dtype == 'datetime64[ns]'
    assert df['date_nan'].dtype == 'datetime64[ns]'
    assert df['date_NA'].dtype == 'datetime64[ns]'
    assert df['date_NaT'].dtype == 'datetime64[ns]'
    assert df['date_NA_specified'].dtype == 'datetime64[ns]'


def test_timedelta_dtype_inference_on_init():
    df = pd.DataFrame({
        'delta_no_nans': (pd.Series([pd.to_datetime('2020-09-01')] * 2) - pd.to_datetime('2020-07-01')),
        'delta_nan': (pd.Series([pd.to_datetime('2020-09-01'), np.nan]) - pd.to_datetime('2020-07-01')),
        'delta_NaT': (pd.Series([pd.to_datetime('2020-09-01'), pd.NaT]) - pd.to_datetime('2020-07-01')),
        'delta_NA_specified': (pd.Series([pd.to_datetime('2020-09-01'), pd.NA], dtype='datetime64[ns]') - pd.to_datetime('2020-07-01')),
    })
    Schema(df)

    assert df['delta_no_nans'].dtype == 'timedelta64[ns]'
    assert df['delta_nan'].dtype == 'timedelta64[ns]'
    assert df['delta_NaT'].dtype == 'timedelta64[ns]'
    assert df['delta_NA_specified'].dtype == 'timedelta64[ns]'


def test_sets_category_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['a', 'b', 'c'], name=column_name),
        pd.Series(['a', None, 'c'], name=column_name),
        pd.Series(['a', np.nan, 'c'], name=column_name),
        pd.Series(['a', pd.NA, 'c'], name=column_name),
        pd.Series(['a', pd.NaT, 'c'], name=column_name),
    ]

    logical_types = [
        Categorical,
        CountryCode,
        Ordinal(order=['a', 'b', 'c']),
        SubRegionCode,
        ZIPCode,
    ]

    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            df = pd.DataFrame(series)
            dt = Schema(df, logical_types=ltypes)
            assert dt.columns[column_name]['logical_type'] == logical_type
            assert dt.columns[column_name]['dtype'] == logical_type.pandas_dtype
            assert df[column_name].dtype == logical_type.pandas_dtype


def test_sets_object_dtype_on_init(latlong_df):
    for column_name in latlong_df.columns:
        ltypes = {
            column_name: LatLong,
        }
        df = latlong_df.loc[:, [column_name]]
        dt = Schema(df, logical_types=ltypes)
        assert dt.columns[column_name]['logical_type'] == LatLong
        assert dt.columns[column_name]['dtype'] == LatLong.pandas_dtype
        assert df[column_name].dtype == LatLong.pandas_dtype


def test_sets_string_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['a', 'b', 'c'], name=column_name),
        pd.Series(['a', None, 'c'], name=column_name),
        pd.Series(['a', np.nan, 'c'], name=column_name),
        pd.Series(['a', pd.NA, 'c'], name=column_name),
    ]

    logical_types = [
        Filepath,
        FullName,
        IPAddress,
        NaturalLanguage,
        PhoneNumber,
        URL,
    ]

    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            df = pd.DataFrame(series)
            dt = Schema(df, logical_types=ltypes)
            assert dt.columns[column_name]['logical_type'] == logical_type
            assert dt.columns[column_name]['dtype'] == logical_type.pandas_dtype
            assert df[column_name].dtype == logical_type.pandas_dtype


def test_sets_boolean_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([True, False, True], name=column_name),
        pd.Series([True, None, True], name=column_name),
        pd.Series([True, np.nan, True], name=column_name),
        pd.Series([True, pd.NA, True], name=column_name),
    ]

    logical_type = Boolean
    for series in series_list:
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        df = pd.DataFrame(series)
        dt = Schema(df, logical_types=ltypes)
        assert dt.columns[column_name]['logical_type'] == logical_type
        assert dt.columns[column_name]['dtype'] == logical_type.pandas_dtype
        assert df[column_name].dtype == logical_type.pandas_dtype


def test_sets_int64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([1, 2, 3], name=column_name),
        pd.Series([1, None, 3], name=column_name),
        pd.Series([1, np.nan, 3], name=column_name),
        pd.Series([1, pd.NA, 3], name=column_name),
    ]

    logical_types = [Integer]
    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            ltypes = {
                column_name: logical_type,
            }
            df = pd.DataFrame(series)
            dt = Schema(df, logical_types=ltypes)
            assert dt.columns[column_name]['logical_type'] == logical_type
            assert dt.columns[column_name]['dtype'] == logical_type.pandas_dtype
            assert df[column_name].dtype == logical_type.pandas_dtype


def test_sets_float64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([1.1, 2, 3], name=column_name),
        pd.Series([1.1, None, 3], name=column_name),
        pd.Series([1.1, np.nan, 3], name=column_name),
    ]

    logical_type = Double
    for series in series_list:
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        df = pd.DataFrame(series)
        dt = Schema(df, logical_types=ltypes)
        assert dt.columns[column_name]['logical_type'] == logical_type
        assert dt.columns[column_name]['dtype'] == logical_type.pandas_dtype
        assert df[column_name].dtype == logical_type.pandas_dtype


def test_sets_datetime64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', None, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', np.nan, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', pd.NA, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', pd.NaT, '2020-01-03'], name=column_name),
    ]

    logical_type = Datetime
    for series in series_list:
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        df = pd.DataFrame(series)
        dt = Schema(df, logical_types=ltypes)
        assert dt.columns[column_name]['logical_type'] == logical_type
        assert dt.columns[column_name]['dtype'] == logical_type.pandas_dtype
        assert df[column_name].dtype == logical_type.pandas_dtype


def test_invalid_dtype_casting():
    column_name = 'test_series'

    # Cannot cast a column with pd.NA to Double
    series = pd.Series([1.1, pd.NA, 3], name=column_name)
    ltypes = {
        column_name: Double,
    }
    err_msg = 'Error converting datatype for column test_series from type object to type ' \
        'float64. Please confirm the underlying data is consistent with logical type Double.'
    with pytest.raises(TypeError, match=err_msg):
        Schema(pd.DataFrame(series), logical_types=ltypes)

    # Cannot cast Datetime to Double
    series = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name)
    ltypes = {
        column_name: Datetime,
    }
    # --> insert when set_types is added
    # dt = Schema(pd.DataFrame(series), logical_types=ltypes)
    # err_msg = 'Error converting datatype for column test_series from type datetime64[ns] to type ' \
    #     'float64. Please confirm the underlying data is consistent with logical type Double.'
    # with pytest.raises(TypeError, match=re.escape(err_msg)):
    #     dt.set_types(logical_types={column_name: Double})

    # Cannot cast invalid strings to integers
    series = pd.Series(['1', 'two', '3'], name=column_name)
    ltypes = {
        column_name: Integer,
    }
    err_msg = 'Error converting datatype for column test_series from type object to type ' \
        'Int64. Please confirm the underlying data is consistent with logical type Integer.'
    with pytest.raises(TypeError, match=err_msg):
        Schema(pd.DataFrame(series), logical_types=ltypes)


def test_schema_init_with_col_descriptions(sample_df):
    descriptions = {
        'age': 'age of the user',
        'signup_date': 'date of account creation'
    }
    dt = Schema(sample_df, column_descriptions=descriptions)
    for name, column in dt.columns.items():
        assert column['description'] == descriptions.get(name)


def test_schema_col_descriptions_warnings(sample_df):
    err_msg = 'column_descriptions must be a dictionary'
    with pytest.raises(TypeError, match=err_msg):
        Schema(sample_df, column_descriptions=34)

    descriptions = {
        'invalid_col': 'not a valid column',
        'signup_date': 'date of account creation'
    }
    err_msg = re.escape("column_descriptions contains columns that are not present in dataframe: ['invalid_col']")
    with pytest.raises(LookupError, match=err_msg):
        Schema(sample_df, column_descriptions=descriptions)


def test_schema_init_with_column_metadata(sample_df):
    column_metadata = {
        'age': {'interesting_values': [33]},
        'signup_date': {'description': 'date of account creation'}
    }
    dt = Schema(sample_df, column_metadata=column_metadata)
    for name, column in dt.columns.items():
        assert column['metadata'] == (column_metadata.get(name) or {})


def test_schema_init_with_metadata(sample_df):
    metadata = {'secondary_time_index': {'is_registered': 'age'}, 'date_created': '11/13/20'}

    dt = Schema(sample_df)
    assert dt.metadata == {}

    dt.metadata = metadata
    assert dt.metadata == metadata

    dt = Schema(sample_df, table_metadata=metadata)  # --> add time index back in
    assert dt.metadata == metadata

    new_data = {'date_created': '1/1/19', 'created_by': 'user1'}
    dt.metadata = {**metadata, **new_data}
    assert dt.metadata == {'secondary_time_index': {'is_registered': 'age'},
                           'date_created': '1/1/19',
                           'created_by': 'user1'}

    dt.metadata.pop('created_by')
    assert dt.metadata == {'secondary_time_index': {'is_registered': 'age'}, 'date_created': '1/1/19'}

    dt.metadata['number'] = 1012034
    assert dt.metadata == {'number': 1012034,
                           'secondary_time_index': {'is_registered': 'age'},
                           'date_created': '1/1/19'}
