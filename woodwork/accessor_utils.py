import pandas as pd

from woodwork.utils import _get_column_logical_type, import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')


def init_series(series, logical_type=None, semantic_tags=None,
                use_standard_tags=True, description=None, metadata=None):
    """Initializes Woodwork typing information for a Series, returning a new Series. The dtype
    of the returned series will be converted to match the dtype associated with the LogicalType.

    Args:
        series (pd.Series, dd.Series, or ks.Series): The original series from which to create
            the Woodwork initialized series.
        logical_type (LogicalType or str, optional): The logical type that should be assigned
            to the series. If no value is provided, the LogicalType for the series will
            be inferred.
        semantic_tags (str or list or set, optional): Semantic tags to assign to the series.
            Defaults to an empty set if not specified. There are two options for
            specifying the semantic tags:
            (str) If only one semantic tag is being set, a single string can be passed.
            (list or set) If multiple tags are being set, a list or set of strings can be passed.
        use_standard_tags (bool, optional): If True, will add standard semantic tags to the series
            based on the inferred or specified logical type of the series. Defaults to True.
        description (str, optional): Optional text describing the contents of the series.
        metadata (dict[str -> json serializable], optional): Metadata associated with the series.

    Returns:
        Series: A series with Woodwork typing information initialized
    """
    logical_type = _get_column_logical_type(series, logical_type, series.name)

    new_series = logical_type.transform(series)
    new_series.ww.init(logical_type=logical_type,
                       semantic_tags=semantic_tags,
                       use_standard_tags=use_standard_tags,
                       description=description,
                       metadata=metadata)
    return new_series


def _is_series(data):
    if isinstance(data, pd.Series):
        return True
    elif dd and isinstance(data, dd.Series):
        return True
    elif ks and isinstance(data, ks.Series):
        return True
    return False


def _is_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return True
    elif dd and isinstance(data, dd.DataFrame):
        return True
    elif ks and isinstance(data, ks.DataFrame):
        return True
    return False


def get_invalid_schema_message(dataframe, schema):
    """Return a message indicating the reason that the provided schema cannot be used to
    initialize Woodwork on the dataframe. If the schema is valid for the dataframe,
    None will be returned.

    Args:
        dataframe (DataFrame): The dataframe against which to check the schema.
        schema (ww.TableSchema): The schema to use in the validity check.

    Returns:
        str or None: The reason that the schema is invalid for the dataframe
    """
    dataframe_cols = set(dataframe.columns)
    schema_cols = set(schema.columns.keys())

    df_cols_not_in_schema = dataframe_cols - schema_cols
    if df_cols_not_in_schema:
        return f'The following columns in the DataFrame were missing from the typing information: '\
            f'{df_cols_not_in_schema}'
    schema_cols_not_in_df = schema_cols - dataframe_cols
    if schema_cols_not_in_df:
        return f'The following columns in the typing information were missing from the DataFrame: '\
            f'{schema_cols_not_in_df}'
    for name in dataframe.columns:
        df_dtype = dataframe[name].dtype
        valid_dtype = schema.logical_types[name]._get_valid_dtype(type(dataframe[name]))
        if str(df_dtype) != valid_dtype:
            return f'dtype mismatch for column {name} between DataFrame dtype, '\
                f'{df_dtype}, and {schema.logical_types[name]} dtype, {valid_dtype}'
    if schema.index is not None and isinstance(dataframe, pd.DataFrame):
        # Index validation not performed for Dask/Koalas
        if not pd.Series(dataframe.index, dtype=dataframe[schema.index].dtype).equals(pd.Series(dataframe[schema.index].values)):
            return 'Index mismatch between DataFrame and typing information'
        elif not dataframe[schema.index].is_unique:
            return 'Index column is not unique'


def is_schema_valid(dataframe, schema):
    """Check if a schema is valid for initializing Woodwork on a dataframe

    Args:
        dataframe (DataFrame): The dataframe against which to check the schema.
        schema (ww.TableSchema): The schema to use in the validity check.

    Returns:
        boolean: Boolean indicating whether the schema is valid for the dataframe
    """

    invalid_schema_message = get_invalid_schema_message(dataframe, schema)
    if invalid_schema_message:
        return False
    return True
