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

# --> not sure if we should go to 6 or 1
SCHEMA_VERSION = '1.0.0'
FORMATS = ['csv', 'pickle', 'parquet']


def typing_info_to_dict(dataframe):
    '''Gets the description for a DataTable, including typing information for each column
    and loading information.
    '''
    # --> confirm schema is initialized
    ordered_columns = dataframe.columns
    column_typing_info = [
        {'name': col_name,
         'ordinal': ordered_columns.get_loc(col_name),
         'logical_type': {
             'parameters': _get_specified_ltype_params(col['logical_type']),
             'type': str(_get_ltype_class(col['logical_type']))
         },
            'physical_type': {
            'type': str(col['dtype'])
         },
            'semantic_tags': sorted(list(col['semantic_tags'])),
            'description': col['description'],
            'metadata': col['metadata']
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
    '''Serialize datatable and write to disk or S3 path.

    Args:
    datatable (DataTable) : Instance of :class:`.DataTable`.
    path (str) : Location on disk to write datatable data and description.
    profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
            Set to False to use an anonymous profile.
    kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method or to specify AWS profile.
    '''
    if _is_s3(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, 'data'))
            dump_table(dataframe, tmpdir, **kwargs)
            file_path = create_archive(tmpdir)

            transport_params = get_transport_params(profile_name)
            use_smartopen(file_path, path, read=False, transport_params=transport_params)
    elif _is_url(path):
        raise ValueError("Writing to URLs is not supported")
    else:
        path = os.path.abspath(path)
        os.makedirs(os.path.join(path, 'data'), exist_ok=True)
        dump_table(dataframe, path, **kwargs)


def dump_table(dataframe, path, **kwargs):
    '''Writes datatable description to table_description.json at the specified path.
    '''
    loading_info = write_dataframe(dataframe, path, **kwargs)

    description = typing_info_to_dict(dataframe)
    description['loading_info'].update(loading_info)

    write_schema(description, path)


def write_schema(description, path):
    # --> add docstring
    try:
        file = os.path.join(path, 'table_description.json')
        with open(file, 'w') as file:
            json.dump(description, file)
    except TypeError:
        raise TypeError('DataTable is not json serializable. Check table and column metadata for values that may not be serializable.')


def write_dataframe(dataframe, path, format='csv', **kwargs):
    '''Write underlying datatable data to disk or S3 path.

    Args:
        datatable (DataTable) : Instance of :class:`.DataTable`.
        path (str) : Location on disk to write datatable data.
        format (str) : Format to use for writing datatable data. Defaults to csv.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method.

    Returns:
        loading_info (dict) : Information on storage location and format of datatable data.
    '''
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
            dataframe = dataframe.copy()
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
        dataframe = dataframe.copy()
        latlong_columns = [col_name for col_name, col in dataframe.ww.columns.items() if _get_ltype_class(col['logical_type']) == ww.logical_types.LatLong]
        dataframe[latlong_columns] = dataframe[latlong_columns].astype(str)

        dataframe.to_parquet(file, **kwargs)
    else:
        error = 'must be one of the following formats: {}'
        raise ValueError(error.format(', '.join(FORMATS)))
    return {'location': location, 'type': format, 'params': kwargs}


def create_archive(tmpdir):
    '''When seralizing to an S3 URL, writes a tar archive.
    '''
    file_name = "ww-{date:%Y-%m-%d_%H%M%S}.tar".format(date=datetime.datetime.now())
    file_path = os.path.join(tmpdir, file_name)
    tar = tarfile.open(str(file_path), 'w')
    tar.add(str(tmpdir) + '/table_description.json', arcname='/table_description.json')
    tar.add(str(tmpdir) + '/data', arcname='/data')
    tar.close()
    return file_path
