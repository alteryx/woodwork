import datetime
import json
import os
import tarfile
import tempfile

from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.utils import (
    _get_ltype_class,
    _get_specified_ltype_params,
    _is_s3,
    _is_url,
    import_or_none
)

dd = import_or_none('dask.dataframe')

SCHEMA_VERSION = '1.0.0'
FORMATS = ['csv', 'pickle', 'parquet']


def datatable_to_metadata(datatable):
    '''Gets the metadata for a DataTable, including typing information for each column
    and loading information.
    '''
    df = datatable.to_dataframe()
    ordered_columns = df.columns
    dt_metadata = [
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
            'semantic_tags': sorted(list(col.semantic_tags))
        }
        for col in datatable.columns.values()
    ]

    if dd and isinstance(df, dd.DataFrame):
        table_type = 'dask'
    else:
        table_type = 'pandas'

    return {
        'schema_version': SCHEMA_VERSION,
        'name': datatable.name,
        'index': datatable.index,
        'time_index': datatable.time_index,
        'metadata': dt_metadata,
        'loading_info': {
            'table_type': table_type
        }

    }


def write_datatable(datatable, path, profile_name=None, **kwargs):
    '''Serialize datatable and write to disk or S3 path.

    Args:
    datatable (DataTable) : Instance of :class:`.DataTable`.
    path (str) : Location on disk to write datatable data and metadata.
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
    '''Writes datatable metadata to table_metadata.json at the specified path.
    '''
    loading_info = write_table_data(datatable, path, **kwargs)

    metadata = datatable_to_metadata(datatable)
    metadata['loading_info'].update(loading_info)

    file = os.path.join(path, 'table_metadata.json')
    with open(file, 'w') as file:
        json.dump(metadata, file)


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
        df.to_csv(
            file,
            index=kwargs['index'],
            sep=kwargs['sep'],
            encoding=kwargs['encoding'],
            compression=kwargs['compression'],
        )
    elif format == 'pickle':
        # Dask currently does not support to_pickle
        if dd and isinstance(df, dd.DataFrame):
            msg = 'Cannot serialize Dask DataTable to pickle'
            raise ValueError(msg)
        df.to_pickle(file, **kwargs)
    elif format == 'parquet':
        # Serializing to parquet format raises an error when columns have object dtype.
        # We currently have no LogicalTypes with physical dtype 'object'
        assert all(dtype != 'object' for dtype in df.dtypes)

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
    tar.add(str(tmpdir) + '/table_metadata.json', arcname='/table_metadata.json')
    tar.add(str(tmpdir) + '/data', arcname='/data')
    tar.close()
    return file_path
