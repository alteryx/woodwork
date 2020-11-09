import json
import os
import tarfile
import tempfile
import warnings
from itertools import zip_longest
from pathlib import Path

import pandas as pd

from woodwork import DataTable
from woodwork.exceptions import OutdatedSchemaWarning, UpgradeSchemaWarning
from woodwork.logical_types import str_to_logical_type
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.serialize import FORMATS, SCHEMA_VERSION
from woodwork.utils import _is_s3, _is_url, import_or_raise


def read_table_metadata(path):
    '''Read DataTable metadata from disk, S3 path, or URL.

        Args:
            path (str): Location on disk, S3 path, or URL to read `table_metadata.json`.

        Returns:
            metadata (dict) : Metadata for :class:`.Datatable`.
    '''

    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    file = os.path.join(path, 'table_metadata.json')
    with open(file, 'r') as file:
        metadata = json.load(file)
    metadata['path'] = path
    return metadata


def metadata_to_datatable(table_metadata, **kwargs):
    '''Deserialize DataTable from table metadata.

    Args:
        table_metadata (dict) : Metadata of a :class:`.DataTable`. Likely generated using :meth:`.serialize.datatable_to_metadata`
        kwargs (keywords): Additional keyword arguments to pass as keywords arguments to the underlying deserialization method.

    Returns:
        datatable (woodwork.DataTable) : Instance of :class:`.DataTable`.
    '''
    _check_schema_version(table_metadata['schema_version'])

    path = table_metadata['path']
    loading_info = table_metadata['loading_info']

    file = os.path.join(path, loading_info['location'])

    load_format = loading_info['type']
    assert load_format in FORMATS

    kwargs = loading_info.get('params', {})
    table_type = loading_info.get('table_type', 'pandas')

    if table_type == 'dask':
        DASK_ERR_MSG = (
            'Cannot load Dask DataTable - unable to import Dask.\n\n'
            'Please install with pip or conda:\n\n'
            'python -m pip install "woodwork[dask]"\n\n'
            'conda install dask'
        )
        lib = import_or_raise('dask.dataframe', DASK_ERR_MSG)
    else:
        lib = pd

    if load_format == 'csv':
        dataframe = lib.read_csv(
            file,
            engine=kwargs['engine'],
            compression=kwargs['compression'],
            encoding=kwargs['encoding'],
        )
    elif load_format == 'pickle':
        dataframe = pd.read_pickle(file, **kwargs)
    elif load_format == 'parquet':
        dataframe = lib.read_parquet(file, engine=kwargs['engine'])

    dtypes = {col['name']: col['physical_type']['type'] for col in table_metadata['metadata']}
    dataframe = dataframe.astype(dtypes)

    logical_types = {}
    semantic_tags = {}
    for col in table_metadata['metadata']:
        col_name = col['name']

        ltype_metadata = col['logical_type']
        ltype = str_to_logical_type(ltype_metadata['type'], params=ltype_metadata['parameters'])

        tags = col['semantic_tags']

        if 'index' in tags:
            tags.remove('index')
        elif 'time_index' in tags:
            tags.remove('time_index')

        logical_types[col_name] = ltype
        semantic_tags[col_name] = tags

    return DataTable(dataframe,
                     name=table_metadata.get('name'),
                     index=table_metadata.get('index'),
                     time_index=table_metadata.get('time_index'),
                     logical_types=logical_types,
                     semantic_tags=semantic_tags,
                     use_standard_tags=False)


def read_datatable(path, profile_name=None, **kwargs):
    '''Read DataTable from disk, S3 path, or URL.

        Args:
            path (str): Directory on disk, S3 path, or URL to read `table_metadata.json`.
            profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
                Set to False to use an anonymous profile.
            kwargs (keywords): Additional keyword arguments to pass as keyword arguments to the underlying deserialization method.
    '''
    if _is_url(path) or _is_s3(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = Path(path).name
            file_path = os.path.join(tmpdir, file_name)
            transport_params = None

            if _is_s3(path):
                transport_params = get_transport_params(profile_name)

            use_smartopen(file_path, path, transport_params)
            with tarfile.open(str(file_path)) as tar:
                tar.extractall(path=tmpdir)

            table_metadata = read_table_metadata(tmpdir)
            return metadata_to_datatable(table_metadata, **kwargs)
    else:
        table_metadata = read_table_metadata(path)
        return metadata_to_datatable(table_metadata, **kwargs)


def _check_schema_version(saved_version_str):
    '''Warns users if the schema used to save their data is greater than the latest
    supported schema or if it is an outdated schema that is no longer supported.
    '''
    saved = saved_version_str.split('.')
    current = SCHEMA_VERSION.split('.')

    for c_num, s_num in zip_longest(current, saved, fillvalue=0):
        if c_num > s_num:
            break
        elif c_num < s_num:
            warnings.warn(UpgradeSchemaWarning().get_warning_message(saved_version_str, SCHEMA_VERSION),
                          UpgradeSchemaWarning)
            break

    # Check if saved has older major version.
    if current[0] > saved[0]:
        warnings.warn(OutdatedSchemaWarning().get_warning_message(saved_version_str),
                      OutdatedSchemaWarning)
