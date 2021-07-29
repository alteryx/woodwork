import re
from inspect import isclass

import numpy as np
import pandas as pd
import pytest
from mock import patch

import woodwork as ww
from woodwork.accessor_utils import (
    _is_dask_dataframe,
    _is_dask_series,
    _is_koalas_dataframe,
    _is_koalas_series,
    init_series
)
from woodwork.exceptions import (
    ColumnNotPresentError,
    IndexTagRemovedWarning,
    ParametersIgnoredWarning,
    TypeConversionError,
    TypingInfoMismatchWarning,
    WoodworkNotInitError
)
from woodwork.logical_types import (
    URL,
    Address,
    Age,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Categorical,
    CountryCode,
    Datetime,
    Double,
    EmailAddress,
    Filepath,
    Integer,
    IntegerNullable,
    IPAddress,
    LatLong,
    NaturalLanguage,
    Ordinal,
    PersonFullName,
    PhoneNumber,
    PostalCode,
    SubRegionCode,
    Unknown
)
from woodwork.table_accessor import (
    WoodworkTableAccessor,
    _check_index,
    _check_logical_types,
    _check_time_index,
    _check_unique_column_names,
    _check_use_standard_tags
)
from woodwork.table_schema import TableSchema
from woodwork.tests.testing_utils import (
    is_property,
    is_public_method,
    to_pandas,
    validate_subset_schema
)
from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def test_check_index_errors(sample_df):
    error_message = 'Specified index column `foo` not found in dataframe'
    with pytest.raises(ColumnNotPresentError, match=error_message):
        _check_index(dataframe=sample_df, index='foo')

    if isinstance(sample_df, pd.DataFrame):
        # Does not check for index uniqueness with Dask
        error_message = 'Index column must be unique'
        with pytest.raises(LookupError, match=error_message):
            _check_index(sample_df, index='age')


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
    with pytest.raises(ColumnNotPresentError, match=error_message):
        _check_logical_types(sample_df, bad_logical_types_keys)


def test_check_time_index_errors(sample_df):
    error_message = 'Specified time index column `foo` not found in dataframe'
    with pytest.raises(ColumnNotPresentError, match=error_message):
        _check_time_index(dataframe=sample_df, time_index='foo')


def test_check_unique_column_names_errors(sample_df):
    if _is_koalas_dataframe(sample_df):
        pytest.skip("Koalas enforces unique column names")
    duplicate_cols_df = sample_df.copy()
    if _is_dask_dataframe(sample_df):
        duplicate_cols_df = dd.concat([duplicate_cols_df, duplicate_cols_df['age']], axis=1)
    else:
        duplicate_cols_df.insert(0, 'age', [18, 21, 65, 43], allow_duplicates=True)
    with pytest.raises(IndexError, match='Dataframe cannot contain duplicate columns names'):
        _check_unique_column_names(duplicate_cols_df)


def test_check_use_standard_tags_errors():
    error_message = 'use_standard_tags must be a dictionary or a boolean'
    with pytest.raises(TypeError, match=error_message):
        _check_use_standard_tags(1)


def test_accessor_init(sample_df):
    assert sample_df.ww.schema is None
    sample_df.ww.init()
    assert isinstance(sample_df.ww.schema, TableSchema)


def test_accessor_schema_property(sample_df):
    sample_df.ww.init()

    assert sample_df.ww._schema is not sample_df.ww.schema
    assert sample_df.ww._schema == sample_df.ww.schema


def test_set_accessor_name(sample_df):
    df = sample_df.copy()
    error = re.escape("Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init")
    with pytest.raises(WoodworkNotInitError, match=error):
        df.ww.name
    with pytest.raises(WoodworkNotInitError, match=error):
        df.ww.name = 'name'

    df.ww.init()
    assert df.ww.name is None
    df.ww.name = 'name'
    assert df.ww.schema.name == 'name'
    assert df.ww.name == 'name'


def test_rename_init_with_name(sample_df):
    df = sample_df.copy()
    df.ww.init(name='name')
    assert df.ww.name == 'name'
    df.ww.name = 'new_name'
    assert df.ww.schema.name == 'new_name'
    assert df.ww.name == 'new_name'


def test_name_error_on_init(sample_df):
    err_msg = "Table name must be a string"
    with pytest.raises(TypeError, match=err_msg):
        sample_df.ww.init(name=123)


def test_name_error_on_update(sample_df):
    sample_df.ww.init()
    err_msg = "Table name must be a string"
    with pytest.raises(TypeError, match=err_msg):
        sample_df.ww.name = 123


def test_name_persists_after_drop(sample_df):
    df = sample_df.copy()
    df.ww.init()

    df.ww.name = 'name'
    assert df.ww.name == 'name'

    dropped_df = df.ww.drop(['id'])
    assert dropped_df.ww.name == 'name'
    assert dropped_df.ww.schema.name == 'name'


def test_set_accessor_metadata(sample_df):
    df = sample_df.copy()
    error = re.escape("Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init")
    with pytest.raises(WoodworkNotInitError, match=error):
        df.ww.metadata
    with pytest.raises(WoodworkNotInitError, match=error):
        df.ww.metadata = {'new': 'metadata'}

    df.ww.init()
    assert df.ww.metadata == {}
    df.ww.metadata = {'new': 'metadata'}
    assert df.ww.schema.metadata == {'new': 'metadata'}
    assert df.ww.metadata == {'new': 'metadata'}


def test_set_metadata_after_init_with_metadata(sample_df):
    df = sample_df.copy()
    df.ww.init(table_metadata={'new': 'metadata'})
    assert df.ww.metadata == {'new': 'metadata'}
    df.ww.metadata = {'new': 'new_metadata'}
    assert df.ww.schema.metadata == {'new': 'new_metadata'}
    assert df.ww.metadata == {'new': 'new_metadata'}


def test_metadata_persists_after_drop(sample_df):
    df = sample_df.copy()
    df.ww.init()

    df.ww.metadata = {'new': 'metadata'}
    assert df.ww.metadata == {'new': 'metadata'}

    dropped_df = df.ww.drop(['id'])
    assert dropped_df.ww.metadata == {'new': 'metadata'}
    assert dropped_df.ww.schema.metadata == {'new': 'metadata'}


def test_metadata_error_on_init(sample_df):
    err_msg = 'Table metadata must be a dictionary.'
    with pytest.raises(TypeError, match=err_msg):
        sample_df.ww.init(table_metadata=123)


def test_metadata_error_on_update(sample_df):
    sample_df.ww.init()
    err_msg = 'Table metadata must be a dictionary.'
    with pytest.raises(TypeError, match=err_msg):
        sample_df.ww.metadata = 123


def test_accessor_physical_types_property(sample_df):
    sample_df.ww.init(logical_types={'age': 'Categorical'})

    assert isinstance(sample_df.ww.physical_types, dict)
    assert set(sample_df.ww.physical_types.keys()) == set(sample_df.columns)
    for k, v in sample_df.ww.physical_types.items():
        logical_type = sample_df.ww.columns[k].logical_type
        if _is_koalas_dataframe(sample_df) and logical_type.backup_dtype is not None:
            assert v == logical_type.backup_dtype
        else:
            assert v == logical_type.primary_dtype


def test_accessor_separation_of_params(sample_df):
    # mix up order of acccessor and schema params
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_name', index='id', semantic_tags={'id': 'test_tag'}, time_index='signup_date')

    assert schema_df.ww.semantic_tags['id'] == {'index', 'test_tag'}
    assert schema_df.ww.index == 'id'
    assert schema_df.ww.time_index == 'signup_date'
    assert schema_df.ww.name == 'test_name'


def test_init_accessor_with_schema(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema', semantic_tags={'id': 'test_tag'}, index='id')
    schema = schema_df.ww._schema

    head_df = schema_df.head(2)
    assert head_df.ww.schema is None
    head_df.ww.init(schema=schema)

    assert head_df.ww._schema is schema
    assert head_df.ww.name == 'test_schema'
    assert head_df.ww.semantic_tags['id'] == {'index', 'test_tag'}

    iloc_df = schema_df.loc[[2, 3]]
    assert iloc_df.ww.schema is None
    iloc_df.ww.init(schema=schema)

    assert iloc_df.ww._schema is schema
    assert iloc_df.ww.name == 'test_schema'
    assert iloc_df.ww.semantic_tags['id'] == {'index', 'test_tag'}
    # Extra parameters do not take effect
    assert isinstance(iloc_df.ww.logical_types['id'], Integer)


def test_accessor_init_errors_methods(sample_df):
    methods_to_exclude = ['init']
    public_methods = [method for method in dir(sample_df.ww) if is_public_method(WoodworkTableAccessor, method)]
    public_methods = [method for method in public_methods if method not in methods_to_exclude]
    method_args_dict = {
        'add_semantic_tags': [{'id': 'new_tag'}],
        'describe': None,
        'pop': ['id'],
        'describe': None,
        'describe_dict': None,
        'drop': ['id'],
        'mutual_information': None,
        'mutual_information_dict': None,
        'remove_semantic_tags': [{'id': 'new_tag'}],
        'rename': [{'id': 'new_id'}],
        'reset_semantic_tags': None,
        'select': [['Double']],
        'set_index': ['id'],
        'set_time_index': ['signup_date'],
        'set_types': [{'id': 'Integer'}],
        'to_disk': ['dir'],
        'to_dictionary': None,
        'value_counts': None,

    }
    error = re.escape("Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init")
    for method in public_methods:
        func = getattr(sample_df.ww, method)
        method_args = method_args_dict[method]
        with pytest.raises(WoodworkNotInitError, match=error):
            if method_args:
                func(*method_args)
            else:
                func()


def test_accessor_init_errors_properties(sample_df):
    props_to_exclude = ['iloc', 'loc', 'schema', '_dataframe']
    props = [prop for prop in dir(sample_df.ww) if is_property(WoodworkTableAccessor, prop) and prop not in props_to_exclude]

    error = re.escape("Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init")
    for prop in props:
        with pytest.raises(WoodworkNotInitError, match=error):
            getattr(sample_df.ww, prop)


def test_init_accessor_with_schema_errors(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init()
    schema = schema_df.ww.schema

    iloc_df = schema_df.iloc[:, :-1]
    assert iloc_df.ww.schema is None

    error = 'Provided schema must be a Woodwork.TableSchema object.'
    with pytest.raises(TypeError, match=error):
        iloc_df.ww.init(schema=int)

    error = ("Woodwork typing information is not valid for this DataFrame: "
             "The following columns in the typing information were missing from the DataFrame: {'datetime_with_NaT'}")
    with pytest.raises(ValueError, match=error):
        iloc_df.ww.init(schema=schema)


def test_accessor_with_schema_parameter_warning(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema', semantic_tags={'id': 'test_tag'}, index='id')
    schema = schema_df.ww.schema

    head_df = schema_df.head(2)

    warning = "A schema was provided and the following parameters were ignored: index, " \
              "time_index, logical_types, already_sorted, use_standard_tags, semantic_tags"
    with pytest.warns(ParametersIgnoredWarning, match=warning):
        head_df.ww.init(index='ignored_id', time_index="ignored_time_index", logical_types={'ignored': 'ltypes'},
                        already_sorted=True, semantic_tags={'ignored_id': 'ignored_test_tag'},
                        use_standard_tags={'id': True, 'age': False}, schema=schema)

    assert head_df.ww.name == 'test_schema'
    assert head_df.ww.semantic_tags['id'] == {'index', 'test_tag'}


def test_accessor_getattr(sample_df):
    schema_df = sample_df.copy()

    # We can access attributes on the Accessor class before the schema is initialized
    assert schema_df.ww.schema is None

    error = re.escape("Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init")
    with pytest.raises(WoodworkNotInitError, match=error):
        schema_df.ww.index

    schema_df.ww.init()

    assert schema_df.ww.name is None
    assert schema_df.ww.index is None
    assert schema_df.ww.time_index is None

    assert set(schema_df.ww.columns.keys()) == set(sample_df.columns)

    error = re.escape("Woodwork has no attribute 'not_present'")
    with pytest.raises(AttributeError, match=error):
        sample_df.ww.init()
        sample_df.ww.not_present


def test_getitem(sample_df):
    df = sample_df
    df.ww.init(
        time_index='signup_date',
        index='id', name='df_name',
        logical_types={'age': 'Double'},
        semantic_tags={'age': {'custom_tag'}},
    )
    assert list(df.columns) == list(df.ww.schema.columns)

    subset = ['id', 'signup_date']
    df_subset = df.ww[subset]
    pd.testing.assert_frame_equal(to_pandas(df[subset]), to_pandas(df_subset))
    assert subset == list(df_subset.ww._schema.columns)
    assert df_subset.ww.index == 'id'
    assert df_subset.ww.time_index == 'signup_date'

    subset = ['age', 'email']
    df_subset = df.ww[subset]
    pd.testing.assert_frame_equal(to_pandas(df[subset]), to_pandas(df_subset))
    assert subset == list(df_subset.ww._schema.columns)
    assert df_subset.ww.index is None
    assert df_subset.ww.time_index is None
    assert isinstance(df_subset.ww.logical_types['age'], Double)
    assert df_subset.ww.semantic_tags['age'] == {'custom_tag', 'numeric'}

    subset = df.ww[[]]
    assert len(subset.ww.columns) == 0
    assert subset.ww.index is None
    assert subset.ww.time_index is None

    series = df.ww['age']
    pd.testing.assert_series_equal(to_pandas(series), to_pandas(df['age']))
    assert isinstance(series.ww.logical_type, Double)
    assert series.ww.semantic_tags == {'custom_tag', 'numeric'}

    series = df.ww['id']
    pd.testing.assert_series_equal(to_pandas(series), to_pandas(df['id']))
    assert isinstance(series.ww.logical_type, Integer)
    assert series.ww.semantic_tags == {'index'}


def test_getitem_init_error(sample_df):
    error = re.escape("Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init")
    with pytest.raises(WoodworkNotInitError, match=error):
        sample_df.ww['age']


def test_getitem_invalid_input(sample_df):
    df = sample_df
    df.ww.init()

    error_msg = r"Column\(s\) '\[1, 2\]' not found in DataFrame"
    with pytest.raises(ColumnNotPresentError, match=error_msg):
        df.ww[['email', 2, 1]]

    error_msg = "Column with name 'invalid_column' not found in DataFrame"
    with pytest.raises(ColumnNotPresentError, match=error_msg):
        df.ww['invalid_column']


def test_accessor_equality(sample_df):
    # Confirm equality with same schema and same data
    schema_df = sample_df.copy()
    schema_df.ww.init()

    copy_df = schema_df.ww.copy()
    assert schema_df.ww == copy_df.ww

    # Confirm not equal with different schema but same data
    copy_df.ww.set_time_index('signup_date')
    assert schema_df.ww != copy_df.ww

    # Confirm not equal with same schema but different data - only pandas
    loc_df = schema_df.ww.loc[:2, :]
    if isinstance(sample_df, pd.DataFrame):
        assert schema_df.ww != loc_df
    else:
        assert schema_df.ww == loc_df


def test_accessor_shallow_equality(sample_df):
    metadata_table = sample_df.copy()
    metadata_table.ww.init(table_metadata={'user': 'user0'})
    diff_metadata_table = sample_df.copy()
    diff_metadata_table.ww.init(table_metadata={'user': 'user2'})

    assert diff_metadata_table.ww.__eq__(metadata_table, deep=False)
    assert not diff_metadata_table.ww.__eq__(metadata_table, deep=True)

    schema = metadata_table.ww.schema
    diff_data_table = metadata_table.ww.loc[:2, :]
    same_data_table = metadata_table.ww.copy()

    assert diff_data_table.ww.schema.__eq__(schema, deep=True)
    assert same_data_table.ww.schema.__eq__(schema, deep=True)

    assert same_data_table.ww.__eq__(metadata_table.ww, deep=False)
    assert same_data_table.ww.__eq__(metadata_table.ww, deep=True)

    assert diff_data_table.ww.__eq__(metadata_table.ww, deep=False)
    if isinstance(sample_df, pd.DataFrame):
        assert not diff_data_table.ww.__eq__(metadata_table.ww, deep=True)


def test_accessor_init_with_valid_string_time_index(time_index_df):
    time_index_df.ww.init(name='schema',
                          index='id',
                          time_index='times')

    assert time_index_df.ww.name == 'schema'
    assert time_index_df.ww.index == 'id'
    assert time_index_df.ww.time_index == 'times'
    assert isinstance(time_index_df.ww.columns[time_index_df.ww.time_index].logical_type, Datetime)


def test_accessor_init_with_numeric_datetime_time_index(time_index_df):
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints', logical_types={'ints': Datetime})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        time_index_df.ww.init(name='schema', time_index='strs', logical_types={'strs': Datetime})

    assert schema_df.ww.time_index == 'ints'
    assert schema_df['ints'].dtype == 'datetime64[ns]'


def test_accessor_with_numeric_time_index(time_index_df):
    # Set a numeric time index on init
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints')
    date_col = schema_df.ww.columns['ints']
    assert schema_df.ww.time_index == 'ints'
    assert isinstance(date_col.logical_type, Integer)
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    # Specify logical type for time index on init
    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='ints', logical_types={'ints': 'Double'})
    date_col = schema_df.ww.columns['ints']
    assert schema_df.ww.time_index == 'ints'
    assert isinstance(date_col.logical_type, Double)
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    schema_df = time_index_df.copy()
    schema_df.ww.init(time_index='strs', logical_types={'strs': 'Double'})
    date_col = schema_df.ww.columns['strs']
    assert schema_df.ww.time_index == 'strs'
    assert isinstance(date_col.logical_type, Double)
    assert date_col.semantic_tags == {'time_index', 'numeric'}

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        time_index_df.ww.init(time_index='ints', logical_types={'ints': 'Categorical'})

    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        time_index_df.ww.init(time_index='letters', logical_types={'strs': 'Integer'})

    # Set numeric time index after init
    schema_df = time_index_df.copy()
    schema_df.ww.init(logical_types={'ints': 'Double'})
    assert schema_df.ww.time_index is None

    schema_df.ww.set_time_index('ints')
    date_col = schema_df.ww.columns['ints']
    assert schema_df.ww.time_index == 'ints'
    assert isinstance(date_col.logical_type, Double)
    assert date_col.semantic_tags == {'numeric', 'time_index'}


def test_numeric_time_index_dtypes(numeric_time_index_df):
    numeric_time_index_df.ww.init(time_index='ints')
    assert numeric_time_index_df.ww.time_index == 'ints'
    assert isinstance(numeric_time_index_df.ww.logical_types['ints'], Integer)
    assert numeric_time_index_df.ww.semantic_tags['ints'] == {'time_index', 'numeric'}

    numeric_time_index_df.ww.set_time_index('floats')
    assert numeric_time_index_df.ww.time_index == 'floats'
    assert isinstance(numeric_time_index_df.ww.logical_types['floats'], Double)
    assert numeric_time_index_df.ww.semantic_tags['floats'] == {'time_index', 'numeric'}

    numeric_time_index_df.ww.set_time_index('with_null')
    assert numeric_time_index_df.ww.time_index == 'with_null'
    assert isinstance(numeric_time_index_df.ww.logical_types['with_null'], IntegerNullable)
    assert numeric_time_index_df.ww.semantic_tags['with_null'] == {'time_index', 'numeric'}


def test_accessor_init_with_invalid_string_time_index(sample_df):
    error_msg = 'Time index column must contain datetime or numeric values'
    with pytest.raises(TypeError, match=error_msg):
        sample_df.ww.init(name='schema', time_index='full_name')


def test_accessor_init_with_string_logical_types(sample_df):
    logical_types = {
        'full_name': 'natural_language',
        'age': 'Double'
    }
    schema_df = sample_df.copy()
    schema_df.ww.init(name='schema',
                      logical_types=logical_types)
    assert isinstance(schema_df.ww.columns['full_name'].logical_type, NaturalLanguage)
    assert isinstance(schema_df.ww.columns['age'].logical_type, Double)

    logical_types = {
        'full_name': 'NaturalLanguage',
        'age': 'IntegerNullable',
        'signup_date': 'Datetime'
    }
    schema_df = sample_df.copy()
    schema_df.ww.init(name='schema',
                      logical_types=logical_types,
                      time_index='signup_date'
                      )
    assert isinstance(schema_df.ww.columns['full_name'].logical_type, NaturalLanguage)
    assert isinstance(schema_df.ww.columns['age'].logical_type, IntegerNullable)
    assert schema_df.ww.time_index == 'signup_date'


def test_int_dtype_inference_on_init():
    df = pd.DataFrame({
        'ints_no_nans': pd.Series([1, 2]),
        'ints_nan': pd.Series([1, np.nan]),
        'ints_NA': pd.Series([1, pd.NA]),
        'ints_NA_specified': pd.Series([1, pd.NA], dtype='Int64')})
    df = df.loc[df.index.repeat(5)].reset_index(drop=True)
    df.ww.init()

    assert df['ints_no_nans'].dtype == 'int64'
    assert df['ints_nan'].dtype == 'float64'
    assert df['ints_NA'].dtype == 'category'
    assert df['ints_NA_specified'].dtype == 'Int64'


def test_bool_dtype_inference_on_init():
    df = pd.DataFrame({
        'bools_no_nans': pd.Series([True, False]),
        'bool_nan': pd.Series([True, np.nan]),
        'bool_NA': pd.Series([True, pd.NA]),
        'bool_NA_specified': pd.Series([True, pd.NA], dtype="boolean")})
    df = df.loc[df.index.repeat(5)].reset_index(drop=True)
    df.ww.init()

    assert df['bools_no_nans'].dtype == 'bool'
    assert df['bool_nan'].dtype == 'category'
    assert df['bool_NA'].dtype == 'category'
    assert df['bool_NA_specified'].dtype == 'boolean'


def test_str_dtype_inference_on_init():
    df = pd.DataFrame({
        'str_no_nans': pd.Series(['a', 'b']),
        'str_nan': pd.Series(['a', np.nan]),
        'str_NA': pd.Series(['a', pd.NA]),
        'str_NA_specified': pd.Series([1, pd.NA], dtype="string"),
    })
    df = df.loc[df.index.repeat(5)].reset_index(drop=True)
    df.ww.init()

    assert df['str_no_nans'].dtype == 'category'
    assert df['str_nan'].dtype == 'category'
    assert df['str_NA'].dtype == 'category'
    assert df['str_NA_specified'].dtype == 'category'


def test_float_dtype_inference_on_init():
    df = pd.DataFrame({
        'floats_no_nans': pd.Series([1.1, 2.2]),
        'floats_nan': pd.Series([1.1, np.nan]),
        'floats_NA': pd.Series([1.1, pd.NA]),
        'floats_nan_specified': pd.Series([1.1, np.nan], dtype='float')})
    df = df.loc[df.index.repeat(5)].reset_index(drop=True)
    df.ww.init()

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
    df.ww.init()

    assert df['date_no_nans'].dtype == 'datetime64[ns]'
    assert df['date_nan'].dtype == 'datetime64[ns]'
    assert df['date_NA'].dtype == 'datetime64[ns]'
    assert df['date_NaT'].dtype == 'datetime64[ns]'
    assert df['date_NA_specified'].dtype == 'datetime64[ns]'


def test_datetime_inference_with_format_param():
    df = pd.DataFrame({
        'index': [0, 1, 2],
        'dates': ["2019/01/01", "2019/01/02", "2019/01/03"],
        'ymd_special': ["2019~01~01", "2019~01~02", "2019~01~03"],
        'mdy_special': pd.Series(['3~11~2000', '3~12~2000', '3~13~2000'], dtype='string'),
    })
    df.ww.init(
        name='df_name',
        logical_types={'ymd_special': Datetime(datetime_format='%Y~%m~%d'),
                       'mdy_special': Datetime(datetime_format='%m~%d~%Y'),
                       'dates': Datetime},
        time_index='ymd_special')

    assert df['dates'].dtype == 'datetime64[ns]'
    assert df['ymd_special'].dtype == 'datetime64[ns]'
    assert df['mdy_special'].dtype == 'datetime64[ns]'

    assert df.ww.time_index == 'ymd_special'
    assert isinstance(df.ww['dates'].ww.logical_type, Datetime)
    assert isinstance(df.ww['ymd_special'].ww.logical_type, Datetime)
    assert isinstance(df.ww['mdy_special'].ww.logical_type, Datetime)

    df.ww.set_time_index('mdy_special')
    assert df.ww.time_index == 'mdy_special'

    df = pd.DataFrame({
        'mdy_special': pd.Series(['3&11&2000', '3&12&2000', '3&13&2000'], dtype='string'),
    })
    df = df.loc[df.index.repeat(5)].reset_index(drop=True)
    df.ww.init()
    assert df['mdy_special'].dtype == 'category'

    df.ww.set_types(logical_types={'mdy_special': Datetime(datetime_format='%m&%d&%Y')})
    assert df['mdy_special'].dtype == 'datetime64[ns]'

    df.ww.set_time_index('mdy_special')
    assert isinstance(df.ww['mdy_special'].ww.logical_type, Datetime)
    assert df.ww.time_index == 'mdy_special'


def test_timedelta_dtype_inference_on_init():
    df = pd.DataFrame({
        'delta_no_nans': (pd.Series([pd.to_datetime('2020-09-01')] * 2) - pd.to_datetime('2020-07-01')),
        'delta_nan': (pd.Series([pd.to_datetime('2020-09-01'), np.nan]) - pd.to_datetime('2020-07-01')),
        'delta_NaT': (pd.Series([pd.to_datetime('2020-09-01'), pd.NaT]) - pd.to_datetime('2020-07-01')),
        'delta_NA_specified': (pd.Series([pd.to_datetime('2020-09-01'), pd.NA], dtype='datetime64[ns]') - pd.to_datetime('2020-07-01')),
    })
    df.ww.init()

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
        PostalCode,
        SubRegionCode
    ]

    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            if isclass(logical_type):
                logical_type = logical_type()

            ltypes = {
                column_name: logical_type,
            }
            df = pd.DataFrame(series)
            df.ww.init(logical_types=ltypes)
            assert df.ww.columns[column_name].logical_type == logical_type
            assert df[column_name].dtype == logical_type.primary_dtype


def test_sets_object_dtype_on_init(latlong_df):
    for column_name in latlong_df.columns:
        ltypes = {
            column_name: LatLong,
        }
        df = latlong_df.loc[:, [column_name]]
        df.ww.init(logical_types=ltypes)
        assert isinstance(df.ww.columns[column_name].logical_type, LatLong)
        assert df[column_name].dtype == LatLong.primary_dtype
        df_pandas = to_pandas(df[column_name])
        expected_val = (3, 4)
        if _is_koalas_dataframe(latlong_df):
            expected_val = [3, 4]
        assert df_pandas.iloc[-1] == expected_val


def test_sets_string_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['a', 'b', 'c'], name=column_name),
        pd.Series(['a', None, 'c'], name=column_name),
        pd.Series(['a', np.nan, 'c'], name=column_name),
        pd.Series(['a', pd.NA, 'c'], name=column_name),
    ]

    logical_types = [
        Address,
        Filepath,
        PersonFullName,
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
            df.ww.init(logical_types=ltypes)
            assert isinstance(df.ww.columns[column_name].logical_type, logical_type)
            assert df[column_name].dtype == logical_type.primary_dtype


def test_sets_boolean_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([True, False, True], name=column_name),
        pd.Series([True, None, True], name=column_name),
        pd.Series([True, np.nan, True], name=column_name),
        pd.Series([True, pd.NA, True], name=column_name),
    ]

    logical_types = [Boolean, BooleanNullable]
    for series in series_list:
        for logical_type in logical_types:
            if series.isnull().any() and logical_type == Boolean:
                continue
            series = series.astype('object')
            ltypes = {
                column_name: logical_type,
            }
            df = pd.DataFrame(series)
            df.ww.init(logical_types=ltypes)
            assert isinstance(df.ww.columns[column_name].logical_type, logical_type)
            assert df[column_name].dtype == logical_type.primary_dtype


def test_sets_int64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series([1, 2, 3], name=column_name),
        pd.Series([1, None, 3], name=column_name),
        pd.Series([1, np.nan, 3], name=column_name),
        pd.Series([1, pd.NA, 3], name=column_name),
    ]

    logical_types = [Integer, IntegerNullable, Age, AgeNullable]
    for series in series_list:
        series = series.astype('object')
        for logical_type in logical_types:
            if series.isnull().any() and logical_type in [Integer, Age]:
                continue
            ltypes = {
                column_name: logical_type,
            }
            df = pd.DataFrame(series)
            df.ww.init(logical_types=ltypes)
            assert isinstance(df.ww.columns[column_name].logical_type, logical_type)
            assert df[column_name].dtype == logical_type.primary_dtype


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
        df.ww.init(logical_types=ltypes)
        assert isinstance(df.ww.columns[column_name].logical_type, logical_type)
        assert df[column_name].dtype == logical_type.primary_dtype


def test_sets_datetime64_dtype_on_init():
    column_name = 'test_series'
    series_list = [
        pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', None, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', np.nan, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', pd.NA, '2020-01-03'], name=column_name),
        pd.Series(['2020-01-01', pd.NaT, '2020-01-03'], name=column_name, dtype='object'),
    ]

    logical_type = Datetime
    for series in series_list:
        series = series.astype('object')
        ltypes = {
            column_name: logical_type,
        }
        df = pd.DataFrame(series)
        df.ww.init(logical_types=ltypes)
        assert isinstance(df.ww.columns[column_name].logical_type, logical_type)
        assert df[column_name].dtype == logical_type.primary_dtype


def test_invalid_dtype_casting():
    column_name = 'test_series'

    # Cannot cast a column with pd.NA to Double
    series = pd.Series([1.1, pd.NA, 3], name=column_name)
    ltypes = {
        column_name: Double,
    }

    err_msg = 'Error converting datatype for test_series from type object to type ' \
        'float64. Please confirm the underlying data is consistent with logical type Double.'
    df = pd.DataFrame(series)
    with pytest.raises(TypeConversionError, match=err_msg):
        df.ww.init(logical_types=ltypes)

    # Cannot cast Datetime to Double
    df = pd.DataFrame({column_name: ['2020-01-01', '2020-01-02', '2020-01-03']})
    df.ww.init(logical_types={column_name: Datetime})

    err_msg = 'Error converting datatype for test_series from type datetime64[ns] to type ' \
        'float64. Please confirm the underlying data is consistent with logical type Double.'
    with pytest.raises(TypeConversionError, match=re.escape(err_msg)):
        df.ww.set_types(logical_types={column_name: Double})

    # Cannot cast invalid strings to integers
    series = pd.Series(['1', 'two', '3'], name=column_name)
    ltypes = {
        column_name: Integer,
    }
    err_msg = 'Error converting datatype for test_series from type object to type ' \
        'int64. Please confirm the underlying data is consistent with logical type Integer.'
    df = pd.DataFrame(series)
    with pytest.raises(TypeConversionError, match=err_msg):
        df.ww.init(logical_types=ltypes)


def test_underlying_index_set_no_index_on_init(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if _is_koalas_dataframe(sample_df):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    input_index = pd.Int64Index([99, 88, 77, 66])

    schema_df = sample_df.copy()
    schema_df.index = input_index.copy()
    pd.testing.assert_index_equal(input_index, schema_df.index)

    schema_df.ww.init()
    assert schema_df.ww.index is None
    pd.testing.assert_index_equal(input_index, schema_df.index)

    sorted_df = schema_df.ww.sort_values('full_name')
    assert sorted_df.ww.index is None
    pd.testing.assert_index_equal(pd.Int64Index([88, 77, 99, 66]), sorted_df.index)


def test_underlying_index_set(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if _is_koalas_dataframe(sample_df):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    # Sets underlying index at init
    schema_df = sample_df.copy()
    schema_df.ww.init(index='full_name')
    assert 'full_name' in schema_df.columns
    assert schema_df.index.name is None
    assert (schema_df.index == schema_df['full_name']).all()

    # Sets underlying index on update
    schema_df = sample_df.copy()
    schema_df.ww.init(index='id')
    schema_df.ww.set_index('full_name')
    assert schema_df.ww.index == 'full_name'
    assert 'full_name' in schema_df.columns
    assert (schema_df.index == schema_df['full_name']).all()
    assert schema_df.index.name is None

    # confirm removing Woodwork index doesn't change underlying index
    schema_df.ww.set_index(None)
    assert schema_df.ww.index is None
    assert (schema_df.index == schema_df['full_name']).all()


def test_underlying_index_reset(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if _is_koalas_dataframe(sample_df):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    specified_index = pd.Index
    unspecified_index = pd.RangeIndex

    sample_df.ww.init()
    assert type(sample_df.index) == unspecified_index

    sample_df.ww.set_index('full_name')
    assert type(sample_df.index) == specified_index

    copied_df = sample_df.ww.copy()
    warning = ("Index mismatch between DataFrame and typing information")
    with pytest.warns(TypingInfoMismatchWarning, match=warning):
        copied_df.ww.reset_index(drop=True, inplace=True)
    assert copied_df.ww.schema is None
    assert type(copied_df.index) == unspecified_index

    sample_df.ww.set_index(None)
    assert type(sample_df.index) == specified_index

    # Use pandas operation to reset index
    reset_df = sample_df.ww.reset_index(drop=True, inplace=False)
    assert type(sample_df.index) == specified_index
    assert type(reset_df.index) == unspecified_index

    sample_df.ww.reset_index(drop=True, inplace=True)
    assert type(sample_df.index) == unspecified_index


def test_underlying_index_unchanged_after_updates(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail('Setting underlying index is not supported with Dask input')
    if _is_koalas_dataframe(sample_df):
        pytest.xfail('Setting underlying index is not supported with Koalas input')

    sample_df.ww.init(index='full_name')
    assert 'full_name' in sample_df
    assert sample_df.ww.index == 'full_name'
    assert (sample_df.index == sample_df['full_name']).all()

    copied_df = sample_df.ww.copy()

    dropped_df = copied_df.ww.drop('full_name')
    assert 'full_name' not in dropped_df
    assert dropped_df.ww.index is None
    assert (dropped_df.index == sample_df['full_name']).all()

    selected_df = copied_df.ww.select('Integer')
    assert 'full_name' not in dropped_df
    assert selected_df.ww.index is None
    assert (selected_df.index == sample_df['full_name']).all()

    iloc_df = copied_df.ww.iloc[:, 2:]
    assert 'full_name' not in iloc_df
    assert iloc_df.ww.index is None
    assert (iloc_df.index == sample_df['full_name']).all()

    loc_df = copied_df.ww.loc[:, ['id', 'email']]
    assert 'full_name' not in loc_df
    assert loc_df.ww.index is None
    assert (loc_df.index == sample_df['full_name']).all()

    subset_df = copied_df.ww[['id', 'email']]
    assert 'full_name' not in subset_df
    assert subset_df.ww.index is None
    assert (subset_df.index == sample_df['full_name']).all()

    reset_tags_df = sample_df.ww.copy()
    reset_tags_df.ww.reset_semantic_tags('full_name', retain_index_tags=False)
    assert reset_tags_df.ww.index is None
    assert (reset_tags_df.index == sample_df['full_name']).all()

    remove_tags_df = sample_df.ww.copy()
    remove_tags_df.ww.remove_semantic_tags({'full_name': 'index'})
    assert remove_tags_df.ww.index is None
    assert (remove_tags_df.index == sample_df['full_name']).all()

    set_types_df = sample_df.ww.copy()
    set_types_df.ww.set_types(semantic_tags={'full_name': 'new_tag'}, retain_index_tags=False)
    assert set_types_df.ww.index is None
    assert (set_types_df.index == sample_df['full_name']).all()

    popped_df = sample_df.ww.copy()
    popped_df.ww.pop('full_name')
    assert popped_df.ww.index is None
    assert (popped_df.index == sample_df['full_name']).all()


def test_accessor_already_sorted(sample_unsorted_df):
    if _is_dask_dataframe(sample_unsorted_df):
        pytest.xfail('Sorting dataframe is not supported with Dask input')
    if _is_koalas_dataframe(sample_unsorted_df):
        pytest.xfail('Sorting dataframe is not supported with Koalas input')

    schema_df = sample_unsorted_df.copy()
    schema_df.ww.init(name='schema',
                      index='id',
                      time_index='signup_date')

    assert schema_df.ww.time_index == 'signup_date'
    assert isinstance(schema_df.ww.columns[schema_df.ww.time_index].logical_type, Datetime)

    sorted_df = to_pandas(sample_unsorted_df).sort_values(['signup_date', 'id']).set_index('id', drop=False)
    sorted_df.index.name = None
    pd.testing.assert_frame_equal(sorted_df,
                                  to_pandas(schema_df), check_index_type=False, check_dtype=False)

    schema_df = sample_unsorted_df.copy()
    schema_df.ww.init(name='schema',
                      index='id',
                      time_index='signup_date',
                      already_sorted=True)

    assert schema_df.ww.time_index == 'signup_date'
    assert isinstance(schema_df.ww.columns[schema_df.ww.time_index].logical_type, Datetime)

    unsorted_df = to_pandas(sample_unsorted_df.set_index('id', drop=False))
    unsorted_df.index.name = None
    pd.testing.assert_frame_equal(unsorted_df, to_pandas(schema_df), check_index_type=False, check_dtype=False)


def test_ordinal_with_order(sample_series):
    if _is_koalas_series(sample_series) or _is_dask_series(sample_series):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not compatible')

    ordinal_with_order = Ordinal(order=['a', 'b', 'c'])
    schema_df = pd.DataFrame(sample_series)
    schema_df.ww.init(logical_types={'sample_series': ordinal_with_order})

    column_logical_type = schema_df.ww.logical_types['sample_series']
    assert isinstance(column_logical_type, Ordinal)
    assert column_logical_type.order == ['a', 'b', 'c']

    schema_df = pd.DataFrame(sample_series)
    schema_df.ww.init()

    schema_df.ww.set_types(logical_types={'sample_series': ordinal_with_order})
    logical_type = schema_df.ww.logical_types['sample_series']
    assert isinstance(logical_type, Ordinal)
    assert logical_type.order == ['a', 'b', 'c']


def test_ordinal_with_incomplete_ranking(sample_series):
    if _is_koalas_series(sample_series) or _is_dask_series(sample_series):
        pytest.xfail('Fails with Dask and Koalas - ordinal data validation not supported')

    ordinal_incomplete_order = Ordinal(order=['a', 'b'])
    error_msg = re.escape("Ordinal column sample_series contains values that are not "
                          "present in the order values provided: ['c']")

    schema_df = pd.DataFrame(sample_series)

    with pytest.raises(ValueError, match=error_msg):
        schema_df.ww.init(logical_types={'sample_series': ordinal_incomplete_order})

    schema_df.ww.init()
    with pytest.raises(ValueError, match=error_msg):
        schema_df.ww.set_types(logical_types={'sample_series': ordinal_incomplete_order})


def test_ordinal_with_nan_values():
    nan_df = pd.DataFrame(pd.Series(['a', 'b', np.nan, 'a'], name='nan_series'))
    ordinal_with_order = Ordinal(order=['a', 'b'])
    nan_df.ww.init(logical_types={'nan_series': ordinal_with_order})

    column_logical_type = nan_df.ww.logical_types['nan_series']
    assert isinstance(column_logical_type, Ordinal)
    assert column_logical_type.order == ['a', 'b']


def test_accessor_with_falsy_column_names(falsy_names_df):
    if _is_dask_dataframe(falsy_names_df):
        pytest.xfail('Dask DataFrames cannot handle integer column names')

    schema_df = falsy_names_df.copy()
    schema_df.ww.init(index=0, time_index='')
    assert schema_df.ww.index == 0
    assert schema_df.ww.time_index == ''

    schema_df.ww.set_time_index(None)
    assert schema_df.ww.time_index is None

    schema_df.ww.set_time_index('')
    assert schema_df.ww.time_index == ''

    popped_col = schema_df.ww.pop('')
    assert '' not in schema_df
    assert '' not in schema_df.ww.columns
    assert schema_df.ww.time_index is None

    schema_df.ww.set_index(None)
    assert schema_df.ww.index is None

    schema_df.ww[''] = popped_col
    assert schema_df.ww[''].name == ''

    renamed_df = schema_df.ww.rename({0: 'col_with_name'})
    assert 0 not in renamed_df.columns
    assert 'col_with_name' in renamed_df.columns


def test_dataframe_methods_on_accessor(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema')

    copied_df = schema_df.ww.copy()

    assert schema_df is not copied_df
    assert schema_df.ww._schema is not copied_df.ww._schema
    assert copied_df.ww.schema == schema_df.ww.schema

    pd.testing.assert_frame_equal(to_pandas(schema_df), to_pandas(copied_df))

    ltype_dtype = 'int64'
    new_dtype = 'string'

    warning = 'Operation performed by astype has invalidated the Woodwork typing information:\n '\
        f'dtype mismatch for column id between DataFrame dtype, {new_dtype}, and Integer dtype, {ltype_dtype}.\n '\
        'Please initialize Woodwork with DataFrame.ww.init'
    with pytest.warns(TypingInfoMismatchWarning, match=warning):
        new_df = schema_df.ww.astype({'id': new_dtype})
    assert new_df['id'].dtype == new_dtype
    assert new_df.ww.schema is None
    assert schema_df.ww.schema is not None


def test_dataframe_methods_on_accessor_new_schema_object(sample_df):
    sample_df.ww.init(index='id', semantic_tags={'email': 'new_tag'},
                      table_metadata={'contributors': ['user1', 'user2'],
                                      'created_on': '2/12/20'},
                      column_metadata={'id': {'important_keys': [1, 2, 3]}})

    copied_df = sample_df.ww.copy()

    assert sample_df.ww.schema == copied_df.ww.schema
    assert sample_df.ww._schema is not copied_df.ww._schema

    copied_df.ww.metadata['contributors'].append('user3')
    assert copied_df.ww.metadata == {'contributors': ['user1', 'user2', 'user3'],
                                     'created_on': '2/12/20'}

    assert sample_df.ww.metadata == {'contributors': ['user1', 'user2'],
                                     'created_on': '2/12/20'}

    copied_df.ww.reset_semantic_tags(retain_index_tags=False)
    assert copied_df.ww.index is None
    assert sample_df.ww.index == 'id'

    assert copied_df.ww.semantic_tags['email'] == set()
    assert sample_df.ww.semantic_tags['email'] == {'new_tag'}

    copied_df.ww.columns['id'].metadata['important_keys'].append(4)
    assert copied_df.ww.columns['id'].metadata == {'important_keys': [1, 2, 3, 4]}
    assert sample_df.ww.columns['id'].metadata == {'important_keys': [1, 2, 3]}


def test_dataframe_methods_on_accessor_inplace(sample_df):
    # TODO: Try to find a supported inplace method for Dask, if one exists
    if _is_dask_dataframe(sample_df):
        pytest.xfail('Dask does not support sort_values or rename inplace.')
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema')

    df_pre_sort = schema_df.copy()

    schema_df.ww.sort_values(['full_name'], inplace=True)
    assert schema_df.ww.name == 'test_schema'

    pd.testing.assert_frame_equal(to_pandas(schema_df), to_pandas(df_pre_sort.sort_values(['full_name'])))

    warning = "Operation performed by insert has invalidated the Woodwork typing information:\n "
    "The following columns in the DataFrame were missing from the typing information: {'new_name'}.\n "
    "Please initialize Woodwork with DataFrame.ww.init"
    with pytest.warns(TypingInfoMismatchWarning, match=warning):
        schema_df.ww.insert(loc=0, column="new_name", value=[1, 2, 3, 4])
    assert 'new_name' in schema_df.columns
    assert schema_df.ww.schema is None


def test_dataframe_methods_on_accessor_returning_series(sample_df):
    schema_df = sample_df[['id', 'age', 'is_registered']]
    schema_df.ww.init(name='test_schema')

    dtypes = schema_df.ww.dtypes

    assert schema_df.ww.name == 'test_schema'
    pd.testing.assert_series_equal(dtypes, schema_df.dtypes)
    all_series = schema_df.ww.all()
    assert schema_df.ww.name == 'test_schema'
    pd.testing.assert_series_equal(to_pandas(all_series), to_pandas(schema_df.all()))


def test_dataframe_methods_on_accessor_other_returns(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='test_schema')

    shape = schema_df.ww.shape

    assert schema_df.ww.name == 'test_schema'
    if _is_dask_dataframe(sample_df):
        shape = (shape[0].compute(), shape[1])
    assert shape == to_pandas(schema_df).shape
    assert schema_df.ww.name == 'test_schema'

    if not _is_dask_dataframe(sample_df):
        # keys() not supported with Dask
        pd.testing.assert_index_equal(schema_df.ww.keys(), schema_df.keys())


def test_dataframe_methods_on_accessor_to_pandas(sample_df):
    if isinstance(sample_df, pd.DataFrame):
        pytest.skip("No need to test converting pandas DataFrame to pandas")

    sample_df.ww.init(name='woodwork', index='id')

    if _is_dask_dataframe(sample_df):
        pd_df = sample_df.ww.compute()
    elif _is_koalas_dataframe(sample_df):
        pd_df = sample_df.ww.to_pandas()
        pytest.skip('Bug #1071: Woodwork not initialized after to_pandas call with Koalas categorical column')
    assert isinstance(pd_df, pd.DataFrame)
    assert pd_df.ww.index == 'id'
    assert pd_df.ww.name == 'woodwork'


def test_get_subset_df_with_schema(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={'full_name': PersonFullName,
                                     'email': EmailAddress,
                                     'phone_number': PhoneNumber,
                                     'age': Double,
                                     'signup_date': Datetime},
                      semantic_tags={'full_name': ['new_tag', 'tag2'],
                                     'age': 'numeric'}
                      )
    schema = schema_df.ww.schema

    empty_df = schema_df.ww._get_subset_df_with_schema([])
    assert len(empty_df.columns) == 0
    assert empty_df.ww.schema is not None
    pd.testing.assert_frame_equal(to_pandas(empty_df), to_pandas(schema_df[[]]))
    validate_subset_schema(empty_df.ww.schema, schema)

    just_index = schema_df.ww._get_subset_df_with_schema(['id'])
    assert just_index.ww.index == schema.index
    assert just_index.ww.time_index is None
    pd.testing.assert_frame_equal(to_pandas(just_index), to_pandas(schema_df[['id']]))
    validate_subset_schema(just_index.ww.schema, schema)

    just_time_index = schema_df.ww._get_subset_df_with_schema(['signup_date'])
    assert just_time_index.ww.time_index == schema.time_index
    assert just_time_index.ww.index is None
    pd.testing.assert_frame_equal(to_pandas(just_time_index), to_pandas(schema_df[['signup_date']]))
    validate_subset_schema(just_time_index.ww.schema, schema)

    transfer_schema = schema_df.ww._get_subset_df_with_schema(['phone_number'])
    assert transfer_schema.ww.index is None
    assert transfer_schema.ww.time_index is None
    pd.testing.assert_frame_equal(to_pandas(transfer_schema), to_pandas(schema_df[['phone_number']]))
    validate_subset_schema(transfer_schema.ww.schema, schema)


def test_select_ltypes_no_match_and_all(sample_df, sample_correct_logical_types):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types=sample_correct_logical_types)

    assert len(schema_df.ww.select(PostalCode).columns) == 0
    assert len(schema_df.ww.select(['PostalCode', PhoneNumber]).columns) == 1

    all_types = ww.type_system.registered_types
    assert len(schema_df.ww.select(exclude=all_types).columns) == 0

    df_all_types = schema_df.ww.select(all_types)
    pd.testing.assert_frame_equal(to_pandas(df_all_types), to_pandas(schema_df))
    assert df_all_types.ww.schema == schema_df.ww.schema


def test_select_ltypes_strings(sample_df, sample_correct_logical_types):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types=sample_correct_logical_types)

    df_multiple_ltypes = schema_df.ww.select(['PersonFullName', 'email_address', 'double', 'BooleanNullable', 'datetime'])
    assert len(df_multiple_ltypes.columns) == 7
    assert 'phone_number' not in df_multiple_ltypes.columns
    assert 'id' not in df_multiple_ltypes.columns

    df_single_ltype = schema_df.ww.select('person_full_name')
    assert set(df_single_ltype.columns) == {'full_name'}


def test_select_ltypes_objects(sample_df, sample_correct_logical_types):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types=sample_correct_logical_types)

    df_multiple_ltypes = schema_df.ww.select([PersonFullName, EmailAddress, Double, BooleanNullable, Datetime])
    assert len(df_multiple_ltypes.columns) == 7
    assert 'phone_number' not in df_multiple_ltypes.columns
    assert 'id' not in df_multiple_ltypes.columns

    df_single_ltype = schema_df.ww.select(PersonFullName)
    assert len(df_single_ltype.columns) == 1


def test_select_ltypes_mixed(sample_df, sample_correct_logical_types):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types=sample_correct_logical_types)

    df_mixed_ltypes = schema_df.ww.select(['PersonFullName', 'email_address', Double])
    assert len(df_mixed_ltypes.columns) == 4
    assert 'phone_number' not in df_mixed_ltypes.columns


def test_select_ltypes_mixed_exclude(sample_df, sample_correct_logical_types):
    schema_df = sample_df.copy()
    schema_df.ww.init(logical_types=sample_correct_logical_types)

    df_mixed_ltypes = schema_df.ww.select(exclude=['PersonFullName', 'email_address', Double])
    assert len(df_mixed_ltypes.columns) == 10
    assert 'full_name' not in df_mixed_ltypes.columns
    assert 'email_address' not in df_mixed_ltypes.columns
    assert 'double' not in df_mixed_ltypes.columns
    assert 'double_with_nan' not in df_mixed_ltypes.columns


def test_select_ltypes_table(sample_df, sample_correct_logical_types):
    schema_df = sample_df.copy()
    schema_df.ww.init(name='testing',
                      index='id',
                      time_index='signup_date',
                      logical_types=sample_correct_logical_types,
                      semantic_tags={'full_name': ['new_tag', 'tag2'],
                                     'age': 'numeric',
                                     })

    df_no_indices = schema_df.ww.select('phone_number')
    assert df_no_indices.ww.index is None
    assert df_no_indices.ww.time_index is None

    df_with_indices = schema_df.ww.select(['Datetime', 'Integer'])
    assert df_with_indices.ww.index == 'id'
    assert df_with_indices.ww.time_index == 'signup_date'

    df_values = schema_df.ww.select(['PersonFullName'])
    assert df_values.ww.name == schema_df.ww.name
    assert df_values.ww.columns['full_name'] == schema_df.ww.columns['full_name']


def test_select_semantic_tags(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(semantic_tags={'full_name': 'tag1',
                                     'email': ['tag2'],
                                     'age': ['numeric', 'tag2'],
                                     'phone_number': ['tag3', 'tag2'],
                                     'is_registered': 'category',
                                     },
                      time_index='signup_date')

    df_one_match = schema_df.ww.select('numeric')
    assert len(df_one_match.columns) == 6
    assert 'age' in df_one_match.columns
    assert 'id' in df_one_match.columns

    df_multiple_matches = schema_df.ww.select('tag2')
    assert len(df_multiple_matches.columns) == 3
    assert 'age' in df_multiple_matches.columns
    assert 'phone_number' in df_multiple_matches.columns
    assert 'email' in df_multiple_matches.columns

    df_multiple_tags = schema_df.ww.select(['numeric', 'time_index'])
    assert len(df_multiple_tags.columns) == 7
    assert 'id' in df_multiple_tags.columns
    assert 'age' in df_multiple_tags.columns
    assert 'signup_date' in df_multiple_tags.columns

    df_overlapping_tags = schema_df.ww.select(['numeric', 'tag2'])
    assert len(df_overlapping_tags.columns) == 8
    assert 'id' in df_overlapping_tags.columns
    assert 'age' in df_overlapping_tags.columns
    assert 'phone_number' in df_overlapping_tags.columns
    assert 'email' in df_overlapping_tags.columns

    df_common_tags = schema_df.ww.select(['category', 'numeric'])
    assert len(df_common_tags.columns) == 8
    assert 'id' in df_common_tags.columns
    assert 'is_registered' in df_common_tags.columns
    assert 'age' in df_common_tags.columns


def test_select_semantic_tags_exclude(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(semantic_tags={'full_name': 'tag1',
                                     'email': ['tag2'],
                                     'age': ['numeric', 'tag2'],
                                     'phone_number': ['tag3', 'tag2'],
                                     'is_registered': 'category',
                                     },
                      time_index='signup_date')

    df_one_match = schema_df.ww.select(exclude='numeric')
    assert len(df_one_match.columns) == 8
    assert 'age' not in df_one_match.columns
    assert 'id' not in df_one_match.columns

    df_multiple_matches = schema_df.ww.select(exclude='tag2')
    assert len(df_multiple_matches.columns) == 11
    assert 'age' not in df_multiple_matches.columns
    assert 'phone_number' not in df_multiple_matches.columns
    assert 'email' not in df_multiple_matches.columns

    df_multiple_tags = schema_df.ww.select(exclude=['numeric', 'time_index'])
    assert len(df_multiple_tags.columns) == 7
    assert 'id' not in df_multiple_tags.columns
    assert 'age' not in df_multiple_tags.columns
    assert 'signup_date' not in df_multiple_tags.columns

    df_overlapping_tags = schema_df.ww.select(exclude=['numeric', 'tag2'])
    assert len(df_overlapping_tags.columns) == 6
    assert 'id' not in df_overlapping_tags.columns
    assert 'age' not in df_overlapping_tags.columns
    assert 'phone_number' not in df_overlapping_tags.columns
    assert 'email' not in df_overlapping_tags.columns

    df_common_tags = schema_df.ww.select(exclude=['category', 'numeric'])
    assert len(df_common_tags.columns) == 6
    assert 'id' not in df_common_tags.columns
    assert 'is_registered' not in df_common_tags.columns
    assert 'age' not in df_common_tags.columns


def test_select_single_inputs(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={
                          'full_name': PersonFullName,
                          'email': EmailAddress,
                          'phone_number': PhoneNumber,
                          'signup_date': Datetime(datetime_format='%Y-%m-%d')
                      },
                      semantic_tags={
                          'full_name': ['new_tag', 'tag2'],
                          'age': 'numeric',
                          'signup_date': 'date_of_birth'
                      })

    df_ltype_string = schema_df.ww.select('person_full_name')
    assert len(df_ltype_string.columns) == 1
    assert 'full_name' in df_ltype_string.columns

    df_ltype_obj = schema_df.ww.select(IntegerNullable)
    assert len(df_ltype_obj.columns) == 2
    assert 'age' in df_ltype_obj.columns

    df_tag_string = schema_df.ww.select('index')
    assert len(df_tag_string.columns) == 1
    assert 'id' in df_tag_string.columns

    df_tag_instantiated = schema_df.ww.select('Datetime')
    assert len(df_tag_instantiated.columns) == 2
    assert 'signup_date' in df_tag_instantiated.columns


def test_select_list_inputs(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={
                          'full_name': PersonFullName,
                          'email': EmailAddress,
                          'phone_number': PhoneNumber,
                          'signup_date': Datetime(datetime_format='%Y-%m-%d'),
                      },
                      semantic_tags={
                          'full_name': ['new_tag', 'tag2'],
                          'age': 'numeric',
                          'signup_date': 'date_of_birth',
                          'email': 'tag2',
                          'is_registered': 'category'
                      })
    df_just_strings = schema_df.ww.select(['PersonFullName', 'index', 'tag2', 'boolean_nullable'])
    assert len(df_just_strings.columns) == 4
    assert 'id' in df_just_strings.columns
    assert 'full_name' in df_just_strings.columns
    assert 'email' in df_just_strings.columns
    assert 'is_registered' in df_just_strings.columns

    df_mixed_selectors = schema_df.ww.select([PersonFullName, 'index', 'time_index', Integer])
    assert len(df_mixed_selectors.columns) == 4
    assert 'id' in df_mixed_selectors.columns
    assert 'full_name' in df_mixed_selectors.columns
    assert 'signup_date' in df_mixed_selectors.columns

    df_common_tags = schema_df.ww.select(['category', 'numeric', BooleanNullable, Datetime])
    assert len(df_common_tags.columns) == 9
    assert 'is_registered' in df_common_tags.columns
    assert 'age' in df_common_tags.columns
    assert 'signup_date' in df_common_tags.columns


def test_select_semantic_tags_no_match(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={
                          'full_name': PersonFullName,
                          'email': EmailAddress,
                          'phone_number': PhoneNumber,
                          'signup_date': Datetime(datetime_format='%Y-%m-%d'),
                      }, semantic_tags={
                          'full_name': ['new_tag', 'tag2'],
                          'age': 'numeric',
                          'signup_date': 'date_of_birth',
                          'email': 'tag2'
                      })

    assert len(schema_df.ww.select(['doesnt_exist']).columns) == 0
    df_multiple_unused = schema_df.ww.select(['doesnt_exist', 'boolean_nullable', 'category', PhoneNumber])
    assert len(df_multiple_unused.columns) == 3

    df_unused_ltype = schema_df.ww.select(['date_of_birth', 'doesnt_exist', PostalCode, Integer])
    assert len(df_unused_ltype.columns) == 3


def test_select_repetitive(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(time_index='signup_date',
                      index='id',
                      name='df_name',
                      logical_types={
                          'full_name': PersonFullName,
                          'email': EmailAddress,
                          'phone_number': PhoneNumber,
                          'signup_date': Datetime(datetime_format='%Y-%m-%d'),
                      }, semantic_tags={
                          'full_name': ['new_tag', 'tag2'],
                          'age': 'numeric',
                          'signup_date': 'date_of_birth',
                          'email': 'tag2'
                      })
    df_repeat_tags = schema_df.ww.select(['new_tag', 'new_tag'])
    assert len(df_repeat_tags.columns) == 1
    assert set(df_repeat_tags.columns) == {'full_name'}

    df_repeat_ltypes = schema_df.ww.select(['PhoneNumber', PhoneNumber, 'phone_number'])
    assert len(df_repeat_ltypes.columns) == 1
    assert set(df_repeat_ltypes.columns) == {'phone_number'}


def test_select_instantiated_ltype():
    ymd_format = Datetime(datetime_format='%Y~%m~%d')

    df = pd.DataFrame({
        'dates': ["2019/01/01", "2019/01/02", "2019/01/03"],
        'ymd': ["2019~01~01", "2019~01~02", "2019~01~03"],
    })
    df.ww.init(
        logical_types={'ymd': ymd_format,
                       'dates': Datetime})

    new_df = df.ww.select('Datetime')
    assert len(new_df.columns) == 2

    new_df = df.ww.select(Datetime)
    assert len(new_df.columns) == 2

    err_msg = "Invalid selector used in include: Datetime cannot be instantiated"
    with pytest.raises(TypeError, match=err_msg):
        df.ww.select(ymd_format)


def test_select_return_schema(sample_df):
    sample_df.ww.init()
    # Multiple column matches
    df_schema = sample_df.ww.select(include='Unknown', return_schema=True)
    assert isinstance(df_schema, TableSchema)
    assert len(df_schema.columns) == 2
    assert df_schema == sample_df.ww.select(include='Unknown').ww.schema

    # Single column match
    single_schema = sample_df.ww.select(include='BooleanNullable', return_schema=True)
    assert isinstance(single_schema, TableSchema)
    assert len(single_schema.columns) == 1
    assert single_schema == sample_df.ww.select(include='BooleanNullable').ww.schema

    # No matches
    empty_schema = sample_df.ww.select(include='IPAddress', return_schema=True)
    assert isinstance(empty_schema, TableSchema)
    assert len(empty_schema.columns) == 0


@pytest.mark.parametrize(
    "ww_type, pandas_type",
    [(["Integer", "IntegerNullable"], "int"),
     (["Double"], "float"),
     (["Datetime"], "datetime"),
     (["Unknown", "EmailAddress"], "string"),
     (["Categorical"], "category"),
     (["Boolean", "BooleanNullable"], "boolean")]
)
def test_select_retains_column_order(ww_type, pandas_type, sample_df):
    if _is_koalas_dataframe(sample_df) and pandas_type in ["category", "string"]:
        pytest.skip("Koalas stores categories as strings")
    sample_df.ww.init()

    ww_schema_column_order = [x for x in sample_df.ww.select(ww_type, return_schema=True).columns.keys()]
    pandas_column_order = [x for x in sample_df.select_dtypes(include=pandas_type).columns]
    assert ww_schema_column_order == pandas_column_order


def test_select_include_and_exclude_error(sample_df):
    sample_df.ww.init()
    err_msg = "Cannot specify values for both 'include' and 'exclude' in a single call."
    with pytest.raises(ValueError, match=err_msg):
        sample_df.ww.select(include='Integer', exclude='Double')

    with pytest.raises(ValueError, match=err_msg):
        sample_df.ww.select(include=[], exclude=[])


def test_select_no_selectors_error(sample_df):
    sample_df.ww.init()
    err_msg = "Must specify values for either 'include' or 'exclude'."
    with pytest.raises(ValueError, match=err_msg):
        sample_df.ww.select()


def test_accessor_set_index(sample_df):
    sample_df.ww.init()

    sample_df.ww.set_index('id')
    assert sample_df.ww.index == 'id'
    if isinstance(sample_df, pd.DataFrame):
        # underlying index not set for Dask/Koalas
        assert (sample_df.index == sample_df['id']).all()

    sample_df.ww.set_index('full_name')
    assert sample_df.ww.index == 'full_name'
    if isinstance(sample_df, pd.DataFrame):
        # underlying index not set for Dask/Koalas
        assert (sample_df.index == sample_df['full_name']).all()

    sample_df.ww.set_index(None)
    assert sample_df.ww.index is None
    if isinstance(sample_df, pd.DataFrame):
        # underlying index not set for Dask/Koalas
        # Check that underlying index doesn't get reset when Woodwork index is removed
        assert (sample_df.index == sample_df['full_name']).all()


def test_accessor_set_index_errors(sample_df):
    sample_df.ww.init()

    error = 'Specified index column `testing` not found in TableSchema.'
    with pytest.raises(ColumnNotPresentError, match=error):
        sample_df.ww.set_index('testing')

    if isinstance(sample_df, pd.DataFrame):
        # Index uniqueness not validate for Dask/Koalas
        error = "Index column must be unique"
        with pytest.raises(LookupError, match=error):
            sample_df.ww.set_index('age')


def test_set_types(sample_df):
    sample_df.ww.init(index='full_name', time_index='signup_date')

    original_df = sample_df.ww.copy()

    sample_df.ww.set_types()
    assert original_df.ww.schema == sample_df.ww.schema
    pd.testing.assert_frame_equal(to_pandas(original_df), to_pandas(sample_df))

    sample_df.ww.set_types(logical_types={'is_registered': 'IntegerNullable'})
    assert sample_df['is_registered'].dtype == 'Int64'

    sample_df.ww.set_types(semantic_tags={'signup_date': ['new_tag']},
                           logical_types={'full_name': 'Categorical'},
                           retain_index_tags=False)
    assert sample_df.ww.index is None
    assert sample_df.ww.time_index is None


def test_set_types_errors(sample_df):
    sample_df.ww.init(index='full_name')

    error = "String invalid is not a valid logical type"
    with pytest.raises(ValueError, match=error):
        sample_df.ww.set_types(logical_types={'id': 'invalid'})

    if isinstance(sample_df, pd.DataFrame):
        # Dask does not error on invalid type conversion until compute
        # Koalas does conversion and fills values with NaN
        error = 'Error converting datatype for email from type string ' \
            'to type float64. Please confirm the underlying data is consistent with ' \
            'logical type Double.'
        with pytest.raises(TypeConversionError, match=error):
            sample_df.ww.set_types(logical_types={'email': 'Double'})

    error = re.escape("Cannot add 'index' tag directly for column email. To set a column as the index, "
                      "use DataFrame.ww.set_index() instead.")
    with pytest.raises(ValueError, match=error):
        sample_df.ww.set_types(semantic_tags={'email': 'index'})


def test_pop(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init(semantic_tags={'age': 'custom_tag'})
    original_schema = schema_df.ww.schema

    popped_series = schema_df.ww.pop('age')

    assert isinstance(popped_series, type(sample_df['age']))
    assert popped_series.ww.semantic_tags == {'custom_tag', 'numeric'}
    pd.testing.assert_series_equal(to_pandas(popped_series), pd.Series([pd.NA, 33, 33, 57], dtype='Int64', name='age'))
    assert isinstance(popped_series.ww.logical_type, IntegerNullable)

    assert 'age' not in schema_df.columns
    assert 'age' not in schema_df.ww.columns

    assert 'age' not in schema_df.ww.logical_types.keys()
    assert 'age' not in schema_df.ww.semantic_tags.keys()

    assert schema_df.ww.schema == original_schema._get_subset_schema(list(schema_df.columns))

    schema_df = sample_df.copy()
    schema_df.ww.init(
        name='table',
        logical_types={'age': IntegerNullable},
        semantic_tags={'age': 'custom_tag'},
        use_standard_tags=False)

    popped_series = schema_df.ww.pop('age')

    assert popped_series.ww.semantic_tags == {'custom_tag'}


def test_pop_index(sample_df):
    sample_df.ww.init(index='id', name='df_name')
    assert sample_df.ww.index == 'id'
    id_col = sample_df.ww.pop('id')
    assert sample_df.ww.index is None
    assert 'index' in id_col.ww.semantic_tags


def test_pop_error(sample_df):
    sample_df.ww.init(
        name='table',
        logical_types={'age': IntegerNullable},
        semantic_tags={'age': 'custom_tag'},
        use_standard_tags=True)

    with pytest.raises(ColumnNotPresentError, match="Column with name 'missing' not found in DataFrame"):
        sample_df.ww.pop("missing")


def test_accessor_drop(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init()

    single_input_df = schema_df.ww.drop('is_registered')
    assert len(single_input_df.ww.columns) == (len(schema_df.columns) - 1)
    assert 'is_registered' not in single_input_df.ww.columns
    assert to_pandas(schema_df).drop('is_registered', axis='columns').equals(to_pandas(single_input_df))

    list_input_df = schema_df.ww.drop(['is_registered'])
    assert len(list_input_df.ww.columns) == (len(schema_df.columns) - 1)
    assert 'is_registered' not in list_input_df.ww.columns
    assert to_pandas(schema_df).drop('is_registered', axis='columns').equals(to_pandas(list_input_df))
    # should be equal to the single input example above
    assert single_input_df.ww.schema == list_input_df.ww.schema
    assert to_pandas(single_input_df).equals(to_pandas(list_input_df))

    multiple_list_df = schema_df.ww.drop(['age', 'full_name', 'is_registered'])
    assert len(multiple_list_df.ww.columns) == (len(schema_df.columns) - 3)
    assert 'is_registered' not in multiple_list_df.ww.columns
    assert 'full_name' not in multiple_list_df.ww.columns
    assert 'age' not in multiple_list_df.ww.columns

    assert to_pandas(schema_df).drop(['is_registered', 'age', 'full_name'], axis='columns').equals(to_pandas(multiple_list_df))

    # Drop the same columns in a different order and confirm resulting DataFrame column order doesn't change
    different_order_df = schema_df.ww.drop(['is_registered', 'age', 'full_name'])
    assert different_order_df.ww.schema == multiple_list_df.ww.schema
    assert to_pandas(multiple_list_df).equals(to_pandas(different_order_df))


def test_accessor_drop_inplace(sample_df):
    sample_df.ww.init()
    inplace_df = sample_df.copy()
    inplace_df.ww.init()

    if _is_dask_dataframe(sample_df):
        error = 'Drop inplace not supported for Dask'
        with pytest.raises(ValueError, match=error):
            inplace_df.ww.drop(['is_registered'], inplace=True)
    elif _is_koalas_dataframe(sample_df):
        error = 'Drop inplace not supported for Koalas'
        with pytest.raises(ValueError, match=error):
            inplace_df.ww.drop(['is_registered'], inplace=True)
    else:
        inplace_df.ww.drop(['is_registered'], inplace=True)
        assert len(inplace_df.ww.columns) == (len(sample_df.columns) - 1)
        assert 'is_registered' not in inplace_df.ww.columns

        assert sample_df.drop('is_registered', axis='columns').equals(inplace_df)


def test_accessor_drop_indices(sample_df):
    sample_df.ww.init(index='id', time_index='signup_date')
    assert sample_df.ww.index == 'id'
    assert sample_df.ww.time_index == 'signup_date'

    dropped_index_df = sample_df.ww.drop('id')
    assert 'id' not in dropped_index_df.ww.columns
    assert dropped_index_df.ww.index is None
    assert dropped_index_df.ww.time_index == 'signup_date'

    dropped_time_index_df = sample_df.ww.drop(['signup_date'])
    assert 'signup_date' not in dropped_time_index_df.ww.columns
    assert dropped_time_index_df.ww.time_index is None
    assert dropped_time_index_df.ww.index == 'id'


def test_accessor_drop_errors(sample_df):
    sample_df.ww.init()

    error = re.escape("Column(s) '['not_present']' not found in DataFrame")
    with pytest.raises(ColumnNotPresentError, match=error):
        sample_df.ww.drop('not_present')

    with pytest.raises(ColumnNotPresentError, match=error):
        sample_df.ww.drop(['age', 'not_present'])

    error = re.escape("Column(s) '['not_present1', 4]' not found in DataFrame")
    with pytest.raises(ColumnNotPresentError, match=error):
        sample_df.ww.drop(['not_present1', 4])


def test_accessor_rename(sample_df):
    table_metadata = {'table_info': 'this is text'}
    id_description = 'the id of the row'
    id_origin = 'base'
    sample_df.ww.init(index='id',
                      time_index='signup_date',
                      table_metadata=table_metadata,
                      column_descriptions={'id': id_description},
                      column_origins={'id': id_origin},
                      semantic_tags={'age': 'test_tag'},
                      logical_types={'age': Double})
    original_df = sample_df.ww.copy()

    new_df = sample_df.ww.rename({'age': 'birthday'})

    assert to_pandas(sample_df.rename(columns={'age': 'birthday'})).equals(to_pandas(new_df))
    # Confirm original dataframe hasn't changed
    assert to_pandas(sample_df).equals(to_pandas(original_df))
    assert sample_df.ww.schema == original_df.ww.schema

    assert original_df.columns.get_loc('age') == new_df.columns.get_loc('birthday')
    pd.testing.assert_series_equal(to_pandas(original_df['age']),
                                   to_pandas(new_df['birthday']),
                                   check_names=False)

    # confirm that metadata and descriptions are there
    assert new_df.ww.metadata == table_metadata
    assert new_df.ww.columns['id'].description == id_description
    assert new_df.ww.columns['id'].origin == id_origin

    old_col = sample_df.ww.columns['age']
    new_col = new_df.ww.columns['birthday']
    assert old_col.logical_type == new_col.logical_type
    assert old_col.semantic_tags == new_col.semantic_tags

    new_df = sample_df.ww.rename({'age': 'full_name', 'full_name': 'age'})

    pd.testing.assert_series_equal(to_pandas(original_df['age']),
                                   to_pandas(new_df['full_name']),
                                   check_names=False)
    pd.testing.assert_series_equal(to_pandas(original_df['full_name']),
                                   to_pandas(new_df['age']),
                                   check_names=False)

    assert original_df.columns.get_loc('age') == new_df.columns.get_loc('full_name')
    assert original_df.columns.get_loc('full_name') == new_df.columns.get_loc('age')


def test_accessor_rename_inplace(sample_df):
    table_metadata = {'table_info': 'this is text'}
    id_description = 'the id of the row'
    id_origin = 'base'
    sample_df.ww.init(index='id',
                      time_index='signup_date',
                      table_metadata=table_metadata,
                      column_descriptions={'id': id_description},
                      column_origins={'id': id_origin},
                      semantic_tags={'age': 'test_tag'},
                      logical_types={'age': Double})
    original_df = sample_df.ww.copy()
    inplace_df = sample_df.ww.copy()

    if _is_dask_dataframe(sample_df):
        error = 'Rename inplace not supported for Dask'
        with pytest.raises(ValueError, match=error):
            inplace_df.ww.rename({'age': 'birthday'}, inplace=True)
    elif _is_koalas_dataframe(sample_df):
        error = 'Rename inplace not supported for Koalas'
        with pytest.raises(ValueError, match=error):
            inplace_df.ww.rename({'age': 'birthday'}, inplace=True)

    else:
        inplace_df.ww.rename({'age': 'birthday'}, inplace=True)

        assert original_df.columns.get_loc('age') == inplace_df.columns.get_loc('birthday')
        pd.testing.assert_series_equal(to_pandas(original_df['age']),
                                       to_pandas(inplace_df['birthday']),
                                       check_names=False)

        # confirm that metadata and descriptions are there
        assert inplace_df.ww.metadata == table_metadata
        assert inplace_df.ww.columns['id'].description == id_description
        assert inplace_df.ww.columns['id'].origin == id_origin

        old_col = sample_df.ww.columns['age']
        new_col = inplace_df.ww.columns['birthday']
        assert old_col.logical_type == new_col.logical_type
        assert old_col.semantic_tags == new_col.semantic_tags

        new_df = sample_df.ww.copy()
        new_df.ww.rename({'age': 'full_name', 'full_name': 'age'}, inplace=True)

        pd.testing.assert_series_equal(to_pandas(original_df['age']),
                                       to_pandas(new_df['full_name']),
                                       check_names=False)
        pd.testing.assert_series_equal(to_pandas(original_df['full_name']),
                                       to_pandas(new_df['age']),
                                       check_names=False)

        assert original_df.columns.get_loc('age') == new_df.columns.get_loc('full_name')
        assert original_df.columns.get_loc('full_name') == new_df.columns.get_loc('age')


def test_accessor_rename_indices(sample_df):
    sample_df.ww.init(
        index='id',
        time_index='signup_date')

    renamed_df = sample_df.ww.rename({'id': 'renamed_index', 'signup_date': 'renamed_time_index'})
    assert 'id' not in renamed_df.columns
    assert 'signup_date' not in renamed_df.columns
    assert 'renamed_index' in renamed_df.columns
    assert 'renamed_time_index' in renamed_df.columns

    if isinstance(sample_df, pd.DataFrame):
        # underlying index not set for Dask/Koalas
        assert all(renamed_df.index == renamed_df['renamed_index'])

    assert renamed_df.ww.index == 'renamed_index'
    assert renamed_df.ww.time_index == 'renamed_time_index'


def test_accessor_schema_properties(sample_df):
    sample_df.ww.init(index='id',
                      time_index='signup_date')

    schema_properties = ['logical_types', 'semantic_tags', 'index', 'time_index', 'use_standard_tags']
    for schema_property in schema_properties:
        prop_from_accessor = getattr(sample_df.ww, schema_property)
        prop_from_schema = getattr(sample_df.ww.schema, schema_property)

        assert prop_from_accessor == prop_from_schema

        # Assumes we don't have setters for any of these attributes
        error = "can't set attribute"
        with pytest.raises(AttributeError, match=error):
            setattr(sample_df.ww, schema_property, 'new_value')


def test_sets_koalas_option_on_init(sample_df_koalas):
    if ks:
        ks.set_option('compute.ops_on_diff_frames', False)
        sample_df_koalas.ww.init()
        assert ks.get_option('compute.ops_on_diff_frames') is True


def test_setitem_invalid_input(sample_df):
    df = sample_df.copy()
    df.ww.init(index='id', time_index='signup_date')

    error_msg = 'New column must be of Series type'
    with pytest.raises(ValueError, match=error_msg):
        df.ww['test'] = [1, 2, 3]

    error_msg = 'Cannot reassign index. Change column name and then use df.ww.set_index to reassign index.'
    with pytest.raises(KeyError, match=error_msg):
        df.ww['id'] = df.id

    error_msg = 'Cannot reassign time index. Change column name and then use df.ww.set_time_index to reassign time index.'
    with pytest.raises(KeyError, match=error_msg):
        df.ww['signup_date'] = df.signup_date


def test_setitem_indexed_column_on_unindexed_dataframe(sample_df):
    sample_df.ww.init()

    col = sample_df.ww.pop('id')
    col.ww.add_semantic_tags(semantic_tags='index')

    warning = 'Cannot add "index" tag on id directly to the DataFrame. The "index" tag has been removed from id. To set this column as a Woodwork index, please use df.ww.set_index'

    with pytest.warns(IndexTagRemovedWarning, match=warning):
        sample_df.ww['id'] = col

    assert sample_df.ww.index is None
    assert ww.is_schema_valid(sample_df, sample_df.ww.schema)
    assert sample_df.ww['id'].ww.semantic_tags == {'numeric'}


def test_setitem_indexed_column_on_indexed_dataframe(sample_df):
    sample_df.ww.init()
    sample_df.ww.set_index('id')

    col = sample_df.ww.pop('id')

    warning = 'Cannot add "index" tag on id directly to the DataFrame. The "index" tag has been removed from id. To set this column as a Woodwork index, please use df.ww.set_index'

    with pytest.warns(IndexTagRemovedWarning, match=warning):
        sample_df.ww['id'] = col

    assert sample_df.ww.index is None
    assert ww.is_schema_valid(sample_df, sample_df.ww.schema)
    assert sample_df.ww['id'].ww.semantic_tags == {'numeric'}

    sample_df.ww.init(logical_types={'email': 'Categorical'})
    sample_df.ww.set_index('id')

    col = sample_df.ww.pop('email')
    col.ww.add_semantic_tags(semantic_tags='index')

    warning = 'Cannot add "index" tag on email directly to the DataFrame. The "index" tag has been removed from email. To set this column as a Woodwork index, please use df.ww.set_index'

    with pytest.warns(IndexTagRemovedWarning, match=warning):
        sample_df.ww['email'] = col
    assert sample_df.ww.index == 'id'
    assert sample_df.ww.semantic_tags['email'] == {'category'}


def test_setitem_indexed_column_on_unindexed_dataframe_no_standard_tags(sample_df):
    sample_df.ww.init()

    col = sample_df.ww.pop('id')
    col.ww.init(semantic_tags='index', use_standard_tags=False)

    warning = 'Cannot add "index" tag on id directly to the DataFrame. The "index" tag has been removed from id. To set this column as a Woodwork index, please use df.ww.set_index'

    with pytest.warns(IndexTagRemovedWarning, match=warning):
        sample_df.ww['id'] = col

    assert sample_df.ww.index is None
    assert ww.is_schema_valid(sample_df, sample_df.ww.schema)
    assert sample_df.ww['id'].ww.semantic_tags == set()


def test_setitem_different_name(sample_df):
    df = sample_df.copy()
    df.ww.init()

    new_series = pd.Series([1, 2, 3, 4], name='wrong', dtype='float')
    if _is_koalas_dataframe(sample_df):
        new_series = ks.Series(new_series)

    # Assign series with name `wrong` to existing column with name `id`
    df.ww['id'] = new_series
    assert df.ww['id'].name == 'id'
    assert 'id' in df.ww.columns
    assert 'wrong' not in df.ww.columns
    assert 'wrong' not in df.columns

    new_series2 = pd.Series([1, 2, 3, 4], name='wrong2', dtype='float')
    if _is_koalas_dataframe(sample_df):
        new_series2 = ks.Series(new_series2)

    # Assign series with name `wrong2` to new column with name `new_col`
    df.ww['new_col'] = new_series2
    assert df.ww['new_col'].name == 'new_col'
    assert 'new_col' in df.ww.columns
    assert 'wrong2' not in df.ww.columns
    assert 'wrong2' not in df.columns


def test_setitem_new_column(sample_df):
    df = sample_df.copy()
    df.ww.init(use_standard_tags=False)

    new_series = pd.Series([1, 2, 3, 4])
    if _is_koalas_dataframe(sample_df):
        new_series = ks.Series(new_series)
    dtype = 'int64'

    df.ww['test_col2'] = new_series
    assert 'test_col2' in df.columns
    assert 'test_col2' in df.ww._schema.columns.keys()
    assert isinstance(df.ww['test_col2'].ww.logical_type, Integer)
    assert df.ww['test_col2'].ww.use_standard_tags is True
    assert df.ww['test_col2'].ww.semantic_tags == {'numeric'}
    assert df.ww['test_col2'].name == 'test_col2'
    assert df.ww['test_col2'].dtype == dtype

    new_series = pd.Series([1, 2, 3], dtype='float')
    if _is_koalas_dataframe(sample_df):
        new_series = ks.Series(new_series)

    new_series = init_series(
        new_series,
        logical_type=Double,
        use_standard_tags=False,
        semantic_tags={'test_tag'},
    )

    df.ww['test_col3'] = new_series
    assert 'test_col3' in df.ww.columns
    assert isinstance(df.ww['test_col3'].ww.logical_type, Double)
    assert df.ww['test_col3'].ww.use_standard_tags is False
    assert df.ww['test_col3'].ww.semantic_tags == {'test_tag'}
    assert df.ww['test_col3'].name == 'test_col3'
    assert df.ww['test_col3'].dtype == 'float'

    # Standard tags and no logical type
    df = sample_df.copy()
    df.ww.init(use_standard_tags=True)

    new_series = pd.Series(['new', 'column', 'inserted'], name='test_col')
    if _is_koalas_dataframe(sample_df):
        dtype = 'string'
        new_series = ks.Series(new_series)
    else:
        dtype = 'category'

    new_series = init_series(new_series, logical_type="Categorical")
    df.ww['test_col'] = new_series
    assert 'test_col' in df.ww.columns
    assert isinstance(df.ww['test_col'].ww.logical_type, Categorical)
    assert df.ww['test_col'].ww.use_standard_tags is True
    assert df.ww['test_col'].ww.semantic_tags == {'category'}
    assert df.ww['test_col'].name == 'test_col'
    assert df.ww['test_col'].dtype == dtype


def test_setitem_overwrite_column(sample_df):
    df = sample_df.copy()
    df.ww.init(
        index='id',
        time_index='signup_date',
        use_standard_tags=True
    )

    # Change to column no change in types
    original_col = df.ww['age']
    new_series = pd.Series([1, 2, 3, None], dtype='Int64')
    if _is_koalas_dataframe(sample_df):
        new_series = ks.Series(new_series)

    dtype = 'Int64'
    new_series = init_series(new_series, use_standard_tags=True)
    df.ww['age'] = new_series

    assert 'age' in df.columns
    assert 'age' in df.ww._schema.columns.keys()
    assert isinstance(df.ww['age'].ww.logical_type, type(original_col.ww.logical_type))
    assert df.ww['age'].ww.semantic_tags == original_col.ww.semantic_tags
    assert df.ww['age'].dtype == dtype
    assert original_col is not df.ww['age']

    # Change dtype, logical types, and tags with conflicting use_standard_tags
    original_col = df['full_name']
    new_series = pd.Series([0, 1, 2], dtype='float')
    if _is_koalas_dataframe(sample_df):
        new_series = ks.Series(new_series)

    new_series = init_series(
        new_series,
        use_standard_tags=False,
        semantic_tags='test_tag',
    )

    df.ww['full_name'] = new_series

    assert 'full_name' in df.columns
    assert 'full_name' in df.ww._schema.columns.keys()
    assert isinstance(df.ww['full_name'].ww.logical_type, Double)
    assert df.ww['full_name'].ww.use_standard_tags is False
    assert df.ww['full_name'].ww.semantic_tags == {'test_tag'}
    assert df.ww['full_name'].dtype == 'float'
    assert original_col is not df.ww['full_name']

    df = sample_df.copy()
    df.ww.init(use_standard_tags=False)

    original_col = df['full_name']
    new_series = pd.Series([0, 1, 2], dtype='float')
    if _is_koalas_dataframe(sample_df):
        new_series = ks.Series(new_series)

    new_series = init_series(
        new_series,
        use_standard_tags=True,
        semantic_tags='test_tag',
    )

    df.ww['full_name'] = new_series

    assert 'full_name' in df.columns
    assert 'full_name' in df.ww._schema.columns.keys()
    assert isinstance(df.ww['full_name'].ww.logical_type, Double)
    assert df.ww['full_name'].ww.use_standard_tags is True
    assert df.ww['full_name'].ww.semantic_tags == {'test_tag', 'numeric'}
    assert df.ww['full_name'].dtype == 'float'
    assert original_col is not df.ww['full_name']


def test_maintain_column_order_on_type_changes(sample_df):
    sample_df.ww.init()
    schema_df = sample_df.ww.copy()

    schema_df.ww.set_types(logical_types={'email': 'Categorical', 'id': 'Double'},
                           semantic_tags={'age': 'tag', 'email': 'tag'})
    assert all(schema_df.columns == sample_df.columns)
    assert all(schema_df.ww.types.index == sample_df.ww.types.index)

    schema_df.ww.set_index('email')
    assert all(schema_df.columns == sample_df.columns)
    assert all(schema_df.ww.types.index == sample_df.ww.types.index)


def test_maintain_column_order_of_dataframe(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init()
    select_df = schema_df.ww.select([Unknown, EmailAddress, Integer, IntegerNullable, Boolean, BooleanNullable, Datetime, Double, Categorical])
    assert all(schema_df.columns == select_df.columns)
    assert all(schema_df.ww.types.index == select_df.ww.types.index)

    renamed_df = schema_df.ww.rename({'email': 'renamed_1', 'id': 'renamed_2'})
    renamed_cols = ['renamed_2', 'full_name', 'renamed_1', 'phone_number', 'age',
                    'signup_date', 'is_registered', 'double', 'double_with_nan', 'integer',
                    'nullable_integer', 'boolean', 'categorical', 'datetime_with_NaT']
    assert all(renamed_cols == renamed_df.columns)
    assert all(renamed_cols == renamed_df.ww.types.index)

    dropped_df = schema_df.ww.drop(['email', 'id', 'is_registered', 'age'])
    cols_left_over = ['full_name', 'phone_number', 'signup_date', 'double', 'double_with_nan',
                      'integer', 'nullable_integer', 'boolean', 'categorical',
                      'datetime_with_NaT']
    assert all(cols_left_over == dropped_df.columns)
    assert all(cols_left_over == dropped_df.ww.types.index)


def test_maintain_column_order_of_input(sample_df):
    schema_df = sample_df.copy()
    schema_df.ww.init()

    reversed_cols = list(schema_df.columns[::-1])

    if not _is_dask_dataframe(sample_df):
        iloc_df = schema_df.ww.iloc[:, list(range(len(schema_df.columns)))[::-1]]
        assert all(reversed_cols == iloc_df.columns)
        assert all(reversed_cols == iloc_df.ww.types.index)

    loc_df = schema_df.ww.loc[:, reversed_cols]
    assert all(reversed_cols == loc_df.columns)
    assert all(reversed_cols == loc_df.ww.types.index)

    getitem_df = schema_df.ww[reversed_cols]
    assert all(reversed_cols == getitem_df.columns)
    assert all(reversed_cols == getitem_df.ww.types.index)


def test_maintain_column_order_disordered_schema(sample_df):
    sample_df.ww.init()
    column_order = list(sample_df.columns)

    scramble_df = sample_df.ww.copy()
    id_col = scramble_df.ww.columns.pop('id')
    scramble_df.ww.columns['id'] = id_col
    assert list(scramble_df.ww.columns.keys()) != column_order

    assert scramble_df.ww.schema == sample_df.ww.schema
    assert all(scramble_df.columns == column_order)
    assert all(scramble_df.ww.types.index == column_order)

    sample_df.ww.init(schema=scramble_df.ww.schema)
    assert all(sample_df.columns == column_order)
    assert all(sample_df.ww.types.index == column_order)


def test_accessor_types(sample_df, sample_inferred_logical_types):
    sample_df.ww.init()

    returned_types = sample_df.ww.types
    assert isinstance(returned_types, pd.DataFrame)
    assert all(returned_types.columns == ['Physical Type', 'Logical Type', 'Semantic Tag(s)'])
    assert returned_types.shape[1] == 3
    assert len(returned_types.index) == len(sample_df.columns)

    correct_physical_types = {name: ltype.primary_dtype for name, ltype in sample_inferred_logical_types.items()}
    if _is_koalas_dataframe(sample_df):
        correct_physical_types['categorical'] = 'string'
    correct_physical_types = pd.Series(list(correct_physical_types.values()),
                                       index=list(correct_physical_types.keys()))

    assert correct_physical_types.equals(returned_types['Physical Type'])
    correct_logical_types = pd.Series([ltype() for ltype in sample_inferred_logical_types.values()],
                                      index=list(sample_inferred_logical_types.keys()))
    assert correct_logical_types.equals(returned_types['Logical Type'])
    correct_semantic_tags = {
        'id': "['numeric']",
        'full_name': "[]",
        'email': "[]",
        'phone_number': "[]",
        'age': "['numeric']",
        'signup_date': "[]",
        'is_registered': "[]",
        'double': "['numeric']",
        'double_with_nan': "['numeric']",
        'integer': "['numeric']",
        'nullable_integer': "['numeric']",
        'boolean': "[]",
        'categorical': "['category']",
        'datetime_with_NaT': "[]",
    }
    correct_semantic_tags = pd.Series(list(correct_semantic_tags.values()),
                                      index=list(correct_semantic_tags.keys()))
    assert correct_semantic_tags.equals(returned_types['Semantic Tag(s)'])


def test_accessor_repr(small_df):
    error = 'Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init'
    with pytest.raises(WoodworkNotInitError, match=error):
        repr(small_df.ww)
    small_df.ww.init()

    table_repr = repr(small_df.ww)
    expected_repr = '                         Physical Type Logical Type Semantic Tag(s)\nColumn                                                             \nsample_datetime_series  datetime64[ns]     Datetime              []'
    assert table_repr == expected_repr

    table_html_repr = small_df.ww._repr_html_()
    expected_repr = '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Physical Type</th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>sample_datetime_series</th>\n      <td>datetime64[ns]</td>\n      <td>Datetime</td>\n      <td>[]</td>\n    </tr>\n  </tbody>\n</table>'
    assert table_html_repr == expected_repr


def test_accessor_repr_empty(empty_df):
    empty_df.ww.init()

    assert repr(empty_df.ww) == 'Empty DataFrame\nColumns: [Physical Type, Logical Type, Semantic Tag(s)]\nIndex: []'
    assert empty_df.ww._repr_html_() == '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Physical Type</th>\n      <th>Logical Type</th>\n      <th>Semantic Tag(s)</th>\n    </tr>\n    <tr>\n      <th>Column</th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n  </tbody>\n</table>'


def test_numeric_column_names(sample_df):
    original_df = sample_df.drop('categorical', axis=1)
    numeric_columns = {col_name: i for col_name, i in zip(original_df.columns, range(0, len(original_df.columns)))}

    original_df.ww.init()
    numeric_via_woodwork = original_df.ww.rename(numeric_columns)

    assert numeric_via_woodwork.ww[0].ww._schema == original_df.ww['id'].ww._schema

    numeric_cols_df = original_df.rename(columns=numeric_columns)
    numeric_cols_df.ww.init()

    assert numeric_cols_df.ww == numeric_via_woodwork.ww

    numeric_cols_df.ww.set_index(0)
    assert numeric_cols_df.ww.index == 0
    assert numeric_cols_df.ww.semantic_tags[0] == {'index'}


@patch("woodwork.table_accessor._validate_accessor_params")
def test_validation_methods_called(mock_validate_accessor_params, sample_df):
    assert not mock_validate_accessor_params.called

    not_validated_df = sample_df.copy()
    not_validated_df.ww.init(validate=False, index='id', logical_types={'age': 'Double'})

    assert not mock_validate_accessor_params.called

    validated_df = sample_df.copy()
    validated_df.ww.init(validate=True, index='id', logical_types={'age': 'Double'})

    assert mock_validate_accessor_params.called

    assert validated_df.ww == not_validated_df.ww
    pd.testing.assert_frame_equal(to_pandas(validated_df), to_pandas(not_validated_df))


def test_maintains_set_logical_type(sample_df):
    if _is_koalas_dataframe(sample_df):
        pytest.xfail("Koalas changed dtype on fillna which invalidates schema")
    sample_df.ww.init(logical_types={'age': 'IntegerNullable'})
    assert isinstance(sample_df.ww.logical_types['age'], IntegerNullable)
    new_df = sample_df.ww.fillna({'age': -1})
    assert isinstance(new_df.ww.logical_types['age'], IntegerNullable)


def test_ltype_conversions_nullable_types():
    df = pd.DataFrame({
        'bool': pd.Series([True, False, True], dtype='bool'),
        'bool_null': pd.Series([True, False, pd.NA], dtype='boolean'),
        'int': pd.Series([1, 7, 3], dtype='int64'),
        'int_null': pd.Series([1, 7, pd.NA], dtype='Int64')
    })

    df.ww.init()
    assert isinstance(df.ww.logical_types['bool'], Boolean)
    assert isinstance(df.ww.logical_types['bool_null'], BooleanNullable)
    assert isinstance(df.ww.logical_types['int'], Integer)
    assert isinstance(df.ww.logical_types['int_null'], IntegerNullable)

    # Test valid conversions
    df.ww.set_types({'bool': 'BooleanNullable', 'int': 'IntegerNullable'})
    assert isinstance(df.ww.logical_types['bool'], BooleanNullable)
    assert isinstance(df.ww.logical_types['int'], IntegerNullable)
    df.ww.set_types({'bool': 'Boolean', 'int': 'Integer'})
    assert isinstance(df.ww.logical_types['bool'], Boolean)
    assert isinstance(df.ww.logical_types['int'], Integer)

    # Test invalid conversions
    error_msg = "Error converting datatype for bool_null from type boolean to type bool. " \
        "Please confirm the underlying data is consistent with logical type Boolean."
    with pytest.raises(TypeConversionError, match=error_msg):
        df.ww.set_types({'bool_null': 'Boolean'})

    error_msg = "Error converting datatype for int_null from type Int64 to type int64. " \
        "Please confirm the underlying data is consistent with logical type Integer."
    with pytest.raises(TypeConversionError, match=error_msg):
        df.ww.set_types({'int_null': 'Integer'})
