import json
import os

import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from woodwork.s3_utils import get_transport_params, use_smartopen_es
from woodwork.utils import _get_ltype_class, _get_ltype_params, _is_s3, _is_url, import_or_raise


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


def metadata_to_datatable(table_metadata, **kwargs):
    '''Deserialize datatable from table metadata.

    Args:
        description (dict) : Description of an :class:`.DataTable`. Likely generated using :meth:`.serialize.datatable_to_metadata`
        kwargs (keywords): Additional keyword arguments to pass as keywords arguments to the underlying deserialization method.

    Returns:
        datatable (DataTable) : Instance of :class:`.DataTable`.
    '''
    pass


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
            return metadata_to_datatable(table_metadata, **kwargs)
    else:
        table_metadata = read_table_metadata(path)
        return metadata_to_datatable(table_metadata, **kwargs)
