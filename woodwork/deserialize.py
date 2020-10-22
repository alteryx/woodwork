import json
import os

import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from woodwork.s3_utils import get_transport_params, use_smartopen_es
from woodwork.utils import _is_s3, _is_url, import_or_raise
from woodwork import DataTable, DataColumn
from woodwork.logical_types import str_to_logical_type


def read_table_metadata(path):
    '''Read datatable metadata from disk, S3 path, or URL.

        Args:
            path (str): Location on disk, S3 path, or URL to read `table_metadata.json`.

        Returns:
            description (dict) : Description of :class:`.Datatable`.
    '''

    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    file = os.path.join(path, 'table_metadata.json')
    with open(file, 'r') as file:
        description = json.load(file)
    description['path'] = path
    return description


def metadata_to_datatable(table_metadata, path, **kwargs):
    '''Deserialize datatable from table metadata.

    Args:
        description (dict) : Description of an :class:`.DataTable`. Likely generated using :meth:`.serialize.datatable_to_metadata`
        path (str) : --> fill in once decided how this is handled
        kwargs (keywords): Additional keyword arguments to pass as keywords arguments to the underlying deserialization method.

    Returns:
        datatable (DataTable) : Instance of :class:`.DataTable`.
    '''
    loading_info = table_metadata['loading_info']
    # check schema version
    # get path
    # There should always be data??
    file = os.path.join(path, loading_info['location'])
    # read table data to get df
    kwargs = loading_info.get('params', {})

    dataframe = pd.read_csv(
        file,
        engine=kwargs['engine'],
        compression=kwargs['compression'],
        encoding=kwargs['encoding'],
    )

    dtypes = {col['name']: col['physical_type']['type'] for col in table_metadata['metadata']}
    dataframe = dataframe.astype(dtypes)

    # --> do i need tto handle latlongs specially?
    logical_types = {}
    semantic_tags = {}
    for col in table_metadata['metadata']:
        col_name = col['name']

        ltype_metadata = col['logical_type']
        #  --> have way of tunring str ltype + p[arams --> instantiated ltype]
        ltype = str_to_logical_type(ltype_metadata['type'], params=ltype_metadata['parameters'])

        tags = col['semantic_tags']

        #  --> not sure this is better than not including them in the list of tags when we serialize
        # We can't have index tags in the datatabe on init
        if 'index' in tags:
            tags.remove('index')
            # --> what happens if you try and set a col as both indx and timei index?
        elif 'time_index' in tags:
            tags.remove('time_index')

        logical_types[col_name] = ltype
        semantic_tags[col_name] = tags

    return DataTable(dataframe,
                     name=table_metadata['name'],
                     index=table_metadata['index'],
                     time_index=table_metadata['time_index'],
                     logical_types=logical_types,
                     semantic_tags=semantic_tags)

    # build dt


def read_datatable(path, profile_name=None, **kwargs):
    '''Read datatable from disk, S3 path, or URL.

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

            use_smartopen_es(file_path, path, transport_params)
            with tarfile.open(str(file_path)) as tar:
                tar.extractall(path=tmpdir)

            table_metadata = read_table_metadata(tmpdir)
            return metadata_to_datatable(table_metadata, file_path, **kwargs)
    else:
        table_metadata = read_table_metadata(path)
        # --> det best way to handle path here whether that's assuming we have the jsoin file type or not
        return metadata_to_datatable(table_metadata, path, **kwargs)
