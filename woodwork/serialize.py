import datetime
import json
import os
import tarfile
import tempfile

import pandas as pd

import woodwork as ww
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.type_sys.utils import (
    _get_ltype_class,
    _get_specified_ltype_params
)
from woodwork.utils import _is_s3, _is_url, import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')

SCHEMA_VERSION = '9.0.0'
FORMATS = ['csv', 'pickle', 'parquet']


def typing_info_to_dict(dataframe):
    """Creates the description for a Woodwork table, including typing information for each column
    and loading information.

    Args:
        dataframe (pd.DataFrame, dd.Dataframe, ks.DataFrame): DataFrame with Woodwork typing
            information initialized.

    Returns:
        dict: Dictionary containing Woodwork typing information
    """
    if dd and isinstance(dataframe, dd.DataFrame):
        # Need to determine the category info for Dask it can be saved below
        category_cols = [colname for colname, col in dataframe.ww._schema.columns.items() if col.is_categorical]
        dataframe = dataframe.ww.categorize(columns=category_cols)
    ordered_columns = dataframe.columns
    column_typing_info = [
        {'name': col_name,
         'ordinal': ordered_columns.get_loc(col_name),
         'use_standard_tags': col.use_standard_tags,
         'logical_type': {
             'parameters': _get_specified_ltype_params(col.logical_type),
             'type': str(_get_ltype_class(col.logical_type))
         },
         'physical_type': {
             'type': str(dataframe[col_name].dtype),
             # Store categorical values so they can be recreated if they are modified during serialization
             'cat_values': dataframe[col_name].dtype.categories.to_list() if str(dataframe[col_name].dtype) == 'category' else None
         },
         'semantic_tags': sorted(list(col.semantic_tags)),
         'description': col.description,
         'metadata': col.metadata,
         }
        for col_name, col in dataframe.ww.columns.items()
    ]

    if dd and isinstance(dataframe, dd.DataFrame):
        table_type = 'dask'
    elif ks and isinstance(dataframe, ks.DataFrame):
        table_type = 'koalas'
    else:
        table_type = 'pandas'

    return {
        'schema_version': SCHEMA_VERSION,
        'name': dataframe.ww.name,
        'index': dataframe.ww.index,
        'time_index': dataframe.ww.time_index,
        'column_typing_info': column_typing_info,
        'loading_info': {
            'table_type': table_type
        },
        'table_metadata': dataframe.ww.metadata
    }


def write_woodwork_table(dataframe, path, profile_name=None, **kwargs):
    """Serialize Woodwork table and write to disk or S3 path.

    Args:
        dataframe (pd.DataFrame, dd.DataFrame, ks.DataFrame): DataFrame with Woodwork typing information initialized.
        path (str) : Location on disk to write the Woodwork table.
        profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
                Set to False to use an anonymous profile.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method or to specify AWS profile.
    """
    if _is_s3(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, 'data'))
            _dump_table(dataframe, tmpdir, **kwargs)
            file_path = _create_archive(tmpdir)

            transport_params = get_transport_params(profile_name)
            use_smartopen(file_path, path, read=False, transport_params=transport_params)
    elif _is_url(path):
        raise ValueError("Writing to URLs is not supported")
    else:
        path = os.path.abspath(path)
        os.makedirs(os.path.join(path, 'data'), exist_ok=True)
        _dump_table(dataframe, path, **kwargs)


def _dump_table(dataframe, path, **kwargs):
    """Writes Woodwork table at the specified path, including both the data and the typing information."""
    loading_info = write_dataframe(dataframe, path, **kwargs)

    typing_info = typing_info_to_dict(dataframe)
    typing_info['loading_info'].update(loading_info)

    write_typing_info(typing_info, path)


def write_typing_info(typing_info, path):
    """Writes Woodwork typing information to the specified path at woodwork_typing_info.json

    Args:
        typing_info (dict): Dictionary containing Woodwork typing information.
    """
    try:
        file = os.path.join(path, 'woodwork_typing_info.json')
        with open(file, 'w') as file:
            json.dump(typing_info, file)
    except TypeError:
        raise TypeError('Woodwork table is not json serializable. Check table and column metadata for values that may not be serializable.')


def write_dataframe(dataframe, path, format='csv', **kwargs):
    """Write underlying DataFrame data to disk or S3 path.

    Args:
        dataframe (pd.DataFrame, dd.DataFrame, ks.DataFrame): DataFrame with Woodwork typing information initialized.
        path (str) : Location on disk to write the Woodwork table.
        format (str) : Format to use for writing Woodwork data. Defaults to csv.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method.

    Returns:
        dict: Information on storage location and format of data.
    """
    format = format.lower()

    ww_name = dataframe.ww.name or 'data'

    if dd and isinstance(dataframe, dd.DataFrame) and format == 'csv':
        basename = "{}-*.{}".format(ww_name, format)
    else:
        basename = '.'.join([ww_name, format])
    location = os.path.join('data', basename)
    file = os.path.join(path, location)

    if format == 'csv':
        compression = kwargs['compression']
        if ks and isinstance(dataframe, ks.DataFrame):
            dataframe = dataframe.ww.copy()
            columns = list(dataframe.select_dtypes('object').columns)
            dataframe[columns] = dataframe[columns].astype(str)
            compression = str(compression)
        dataframe.to_csv(
            file,
            index=kwargs['index'],
            sep=kwargs['sep'],
            encoding=kwargs['encoding'],
            compression=compression
        )
    elif format == 'pickle':
        # Dask and Koalas currently do not support to_pickle
        if not isinstance(dataframe, pd.DataFrame):
            msg = 'DataFrame type not compatible with pickle serialization. Please serialize to another format.'
            raise ValueError(msg)
        dataframe.to_pickle(file, **kwargs)
    elif format == 'parquet':
        # Latlong columns in pandas and Dask DataFrames contain tuples, which raises
        # an error in parquet format.
        dataframe = dataframe.ww.copy()
        latlong_columns = [col_name for col_name, col in dataframe.ww.columns.items() if _get_ltype_class(col.logical_type) == ww.logical_types.LatLong]
        dataframe[latlong_columns] = dataframe[latlong_columns].astype(str)

        dataframe.to_parquet(file, **kwargs)
    else:
        error = 'must be one of the following formats: {}'
        raise ValueError(error.format(', '.join(FORMATS)))
    return {'location': location, 'type': format, 'params': kwargs}


def _create_archive(tmpdir):
    """When seralizing to an S3 URL, writes a tar archive."""
    file_name = "ww-{date:%Y-%m-%d_%H%M%S}.tar".format(date=datetime.datetime.now())
    file_path = os.path.join(tmpdir, file_name)
    tar = tarfile.open(str(file_path), 'w')
    tar.add(str(tmpdir) + '/woodwork_typing_info.json', arcname='/woodwork_typing_info.json')
    tar.add(str(tmpdir) + '/data', arcname='/data')
    tar.close()
    return file_path
