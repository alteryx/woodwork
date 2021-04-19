import json
import os
import tarfile
import tempfile
import warnings
from itertools import zip_longest
from pathlib import Path

import pandas as pd

import woodwork as ww
from woodwork.exceptions import OutdatedSchemaWarning, UpgradeSchemaWarning
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.serialize import FORMATS, SCHEMA_VERSION
from woodwork.utils import _is_s3, _is_url, import_or_none, import_or_raise

ks = import_or_none('databricks.koalas')


def read_table_typing_information(path):
    """Read Woodwork typing information from disk, S3 path, or URL.

        Args:
            path (str): Location on disk, S3 path, or URL to read `woodwork_typing_info.json`.

        Returns:
            dict: Woodwork typing information dictionary
    """
    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    file = os.path.join(path, 'woodwork_typing_info.json')
    with open(file, 'r') as file:
        typing_info = json.load(file)
    typing_info['path'] = path
    return typing_info


def _typing_information_to_woodwork_table(table_typing_info, validate, **kwargs):
    """Deserialize Woodwork table from table description.

    Args:
        table_typing_info (dict) : Woodwork typing information. Likely generated using :meth:`.serialize.typing_info_to_dict`
        validate (bool): Whether parameter and data validation should occur during table initialization
        kwargs (keywords): Additional keyword arguments to pass as keywords arguments to the underlying deserialization method.

    Returns:
        DataFrame: DataFrame with Woodwork typing information initialized.
    """
    _check_schema_version(table_typing_info['schema_version'])

    path = table_typing_info['path']
    loading_info = table_typing_info['loading_info']

    file = os.path.join(path, loading_info['location'])

    load_format = loading_info['type']
    assert load_format in FORMATS

    kwargs = loading_info.get('params', {})
    table_type = loading_info.get('table_type', 'pandas')

    logical_types = {}
    semantic_tags = {}
    column_descriptions = {}
    column_metadata = {}
    use_standard_tags = {}
    dtypes = {}
    for col in table_typing_info['column_typing_info']:
        col_name = col['name']

        ltype_metadata = col['logical_type']
        ltype = ww.type_system.str_to_logical_type(ltype_metadata['type'], params=ltype_metadata['parameters'])

        tags = col['semantic_tags']

        if 'index' in tags:
            tags.remove('index')
        elif 'time_index' in tags:
            tags.remove('time_index')

        logical_types[col_name] = ltype
        semantic_tags[col_name] = tags
        column_descriptions[col_name] = col['description']
        column_metadata[col_name] = col['metadata']
        use_standard_tags[col_name] = col['use_standard_tags']

        if col['physical_type']['type'] == 'category':
            # Make sure categories are recreated properly
            cat_values = col['physical_type']['cat_values']
            if table_type == 'pandas' and pd.__version__ > '1.1.5':
                cat_object = pd.CategoricalDtype(pd.Index(cat_values, dtype='object'))
            else:
                cat_object = pd.CategoricalDtype(pd.Series(cat_values))
            dtypes[col_name] = cat_object
        # elif not (ks and col['physical_type']['type'] == 'object'):
        #     # Can't specify `object` for koalas
        #     dtypes[col_name] = col['physical_type']['type']

    compression = kwargs['compression']
    if table_type == 'dask':
        DASK_ERR_MSG = (
            'Cannot load Dask DataFrame - unable to import Dask.\n\n'
            'Please install with pip or conda:\n\n'
            'python -m pip install "woodwork[dask]"\n\n'
            'conda install dask'
        )
        lib = import_or_raise('dask.dataframe', DASK_ERR_MSG)
    elif table_type == 'koalas':
        KOALAS_ERR_MSG = (
            'Cannot load Koalas DataFrame - unable to import Koalas.\n\n'
            'Please install with pip or conda:\n\n'
            'python -m pip install "woodwork[koalas]"\n\n'
            'conda install koalas\n\n'
            'conda install pyspark'
        )
        lib = import_or_raise('databricks.koalas', KOALAS_ERR_MSG)
        compression = str(compression)
    else:
        lib = pd

    if load_format == 'csv':
        dataframe = lib.read_csv(
            file,
            engine=kwargs['engine'],
            compression=compression,
            encoding=kwargs['encoding'],
            dtype=dtypes,
        )
    elif load_format == 'pickle':
        dataframe = pd.read_pickle(file, **kwargs)
    elif load_format == 'parquet':
        dataframe = lib.read_parquet(file, engine=kwargs['engine'])

    dataframe.ww.init(
        name=table_typing_info.get('name'),
        index=table_typing_info.get('index'),
        time_index=table_typing_info.get('time_index'),
        logical_types=logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=use_standard_tags,
        table_metadata=table_typing_info.get('table_metadata'),
        column_metadata=column_metadata,
        column_descriptions=column_descriptions,
        validate=validate)

    return dataframe


def read_woodwork_table(path, profile_name=None, validate=False, **kwargs):
    """Read Woodwork table from disk, S3 path, or URL.

        Args:
            path (str): Directory on disk, S3 path, or URL to read `woodwork_typing_info.json`.
            profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
                Set to False to use an anonymous profile.
            validate (bool, optional): Whether parameter and data validation should occur when initializing Woodwork dataframe
                during deserialization. Defaults to False. Note: If serialized data was modified outside of Woodwork and you
                are unsure of the validity of the data or typing information, `validate` should be set to True.
            kwargs (keywords): Additional keyword arguments to pass as keyword arguments to the underlying deserialization method.

        Returns:
            DataFrame: DataFrame with Woodwork typing information initialized.
    """
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

            table_typing_info = read_table_typing_information(tmpdir)
            return _typing_information_to_woodwork_table(table_typing_info, validate, **kwargs)
    else:
        table_typing_info = read_table_typing_information(path)
        return _typing_information_to_woodwork_table(table_typing_info, validate, **kwargs)


def _check_schema_version(saved_version_str):
    """Warns users if the schema used to save their data is greater than the latest
    supported schema or if it is an outdated schema that is no longer supported."""
    saved = saved_version_str.split('.')
    current = SCHEMA_VERSION.split('.')

    for c_num, s_num in zip_longest(current, saved, fillvalue=0):
        if int(c_num) > int(s_num):
            break
        elif int(c_num) < int(s_num):
            warnings.warn(UpgradeSchemaWarning().get_warning_message(saved_version_str, SCHEMA_VERSION),
                          UpgradeSchemaWarning)
            break

    # Check if saved has older major version.
    if current[0] > saved[0]:
        warnings.warn(OutdatedSchemaWarning().get_warning_message(saved_version_str),
                      OutdatedSchemaWarning)
