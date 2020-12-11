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

SCHEMA_VERSION = '5.0.0'
FORMATS = ['csv', 'pickle', 'parquet']


def datatable_to_description(datatable):
    '''Gets the description for a DataTable, including typing information for each column
    and loading information.
    '''
    df = datatable.to_dataframe()
    ordered_columns = df.columns
    column_metadata = [
        {
            'name': col.name,
            'ordinal': ordered_columns.get_loc(col.name),
            'logical_type': {
                'parameters': _get_specified_ltype_params(col.logical_type),
                'type': str(_get_ltype_class(col.logical_type))
            },
            'physical_type': {
                'type': str(col.dtype)
            },
            'semantic_tags': sorted(list(col.semantic_tags)),
            'description': col.description,
            'metadata': col.metadata
        }
        for col in datatable.columns.values()
    ]

    if dd and isinstance(df, dd.DataFrame):
        table_type = 'dask'
    elif ks and isinstance(df, ks.DataFrame):
        table_type = 'koalas'
    else:
        table_type = 'pandas'

    return {
        'schema_version': SCHEMA_VERSION,
        'name': datatable.name,
        'index': datatable.index,
        'time_index': datatable.time_index,
        'column_metadata': column_metadata,
        'loading_info': {
            'table_type': table_type
        },
        'table_metadata': datatable.metadata
    }


def write_datatable(datatable, path, profile_name=None, **kwargs):
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
            dump_table(datatable, tmpdir, **kwargs)
            file_path = create_archive(tmpdir)

            transport_params = get_transport_params(profile_name)
            use_smartopen(file_path, path, read=False, transport_params=transport_params)
    elif _is_url(path):
        raise ValueError("Writing to URLs is not supported")
    else:
        path = os.path.abspath(path)
        os.makedirs(os.path.join(path, 'data'), exist_ok=True)
        dump_table(datatable, path, **kwargs)


def dump_table(datatable, path, **kwargs):
    '''Writes datatable description to table_description.json at the specified path.
    '''
    loading_info = write_table_data(datatable, path, **kwargs)
    description = datatable_to_description(datatable)
    description['loading_info'].update(loading_info)

    try:
        file = os.path.join(path, 'table_description.json')
        with open(file, 'w') as file:
            json.dump(description, file)
    except TypeError as e:
        raise TypeError('DataTable is not json serializable: ' + e.args[0].replace("'", ""))


def write_table_data(datatable, path, format='csv', **kwargs):
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

    dt_name = datatable.name or 'data'
    df = datatable.to_dataframe()

    if dd and isinstance(df, dd.DataFrame) and format == 'csv':
        basename = "{}-*.{}".format(dt_name, format)
    else:
        basename = '.'.join([dt_name, format])
    location = os.path.join('data', basename)
    file = os.path.join(path, location)

    if format == 'csv':
        compression = kwargs['compression']
        if ks and isinstance(df, ks.DataFrame):
            df = df.copy()
            columns = list(df.select_dtypes('object').columns)
            df[columns] = df[columns].astype(str)
            compression = str(compression)
        df.to_csv(
            file,
            index=kwargs['index'],
            sep=kwargs['sep'],
            encoding=kwargs['encoding'],
            compression=compression
        )
    elif format == 'pickle':
        # Dask and Koalas currently do not support to_pickle
        if not isinstance(df, pd.DataFrame):
            msg = 'DataFrame type not compatible with pickle serialization. Please serialize to another format.'
            raise ValueError(msg)
        df.to_pickle(file, **kwargs)
    elif format == 'parquet':
        # Latlong columns in pandas and Dask DataFrames contain tuples, which raises
        # an error in parquet format.
        df = df.copy()
        latlong_columns = [col_name for col_name, col in datatable.columns.items() if _get_ltype_class(col.logical_type) == ww.logical_types.LatLong]
        df[latlong_columns] = df[latlong_columns].astype(str)

        df.to_parquet(file, **kwargs)
    else:
        error = 'must be one of the following formats: {}'
        raise ValueError(error.format(', '.join(FORMATS)))
    return {'location': location, 'type': format, 'params': kwargs}


def create_archive(tmpdir):
    '''When seralizing to an S3 URL, writes a tar archive.
    '''
    file_name = "dt-{date:%Y-%m-%d_%H%M%S}.tar".format(date=datetime.datetime.now())
    file_path = os.path.join(tmpdir, file_name)
    tar = tarfile.open(str(file_path), 'w')
    tar.add(str(tmpdir) + '/table_description.json', arcname='/table_description.json')
    tar.add(str(tmpdir) + '/data', arcname='/data')
    tar.close()
    return file_path
