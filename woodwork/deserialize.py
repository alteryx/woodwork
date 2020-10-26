import json
import os
import tarfile
import tempfile
import warnings
from itertools import zip_longest
from pathlib import Path

import pandas as pd

from woodwork import DataTable
from woodwork.logical_types import str_to_logical_type
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.serialize import SCHEMA_VERSION
from woodwork.utils import _is_s3, _is_url


def read_table_metadata(path):
    '''Read datatable metadata from disk, S3 path, or URL.

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
    '''Deserialize datatable from table metadata.

    Args:
        description (dict) : Description of an :class:`.DataTable`. Likely generated using :meth:`.serialize.datatable_to_metadata`
        path (str) : Directory from which to read table_metadata.json
        kwargs (keywords): Additional keyword arguments to pass as keywords arguments to the underlying deserialization method.

    Returns:
        datatable (DataTable) : Instance of :class:`.DataTable`.
    '''
    _check_schema_version(table_metadata)

    path = table_metadata['path']
    loading_info = table_metadata['loading_info']
    file = os.path.join(path, loading_info['location'])
    kwargs = loading_info.get('params', {})

    dataframe = pd.read_csv(
        file,
        engine=kwargs['engine'],
        compression=kwargs['compression'],
        encoding=kwargs['encoding'],
    )

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
                     name=table_metadata['name'],
                     index=table_metadata['index'],
                     time_index=table_metadata['time_index'],
                     logical_types=logical_types,
                     semantic_tags=semantic_tags)


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

            use_smartopen(file_path, path, transport_params)
            with tarfile.open(str(file_path)) as tar:
                tar.extractall(path=tmpdir)

            table_metadata = read_table_metadata(tmpdir)
            return metadata_to_datatable(table_metadata, **kwargs)
    else:
        table_metadata = read_table_metadata(path)
        return metadata_to_datatable(table_metadata, **kwargs)


def _check_schema_version(metadata):
    # --> see where it might be better to use .get
    saved_version_str = metadata['schema_version']
    saved = saved_version_str.split('.')
    current = SCHEMA_VERSION.split('.')

    warning_text_upgrade = ('The schema version of the saved ww.DataTable'
                            '(%s) is greater than the latest supported (%s). '
                            'You may need to upgrade featuretools. Attempting to load ww.DataTable ...'
                            % (saved_version_str, SCHEMA_VERSION))

    for c_num, s_num in zip_longest(current, saved, fillvalue=0):
        if c_num > s_num:
            break
        elif c_num < s_num:
            warnings.warn(warning_text_upgrade)
            break

    warning_text_outdated = ('The schema version of the saved ww.DataTable '
                             '(%s) is no longer supported by this version '
                             'of featuretools. Attempting to load ww.DataTable ...'
                             % (saved_version_str))
    # Check if saved has older major version.
    if current[0] > saved[0]:
        warnings.warn(warning_text_outdated)
