import datetime
import json
import os
import tarfile
import tempfile

from .version import __version__

from woodwork.s3_utils import get_transport_params, use_smartopen_es
from woodwork.utils import _get_ltype_class, _get_ltype_params, _is_s3, _is_url

FEATURETOOLS_VERSION = '0.20.0'  # --> shouldnt have to hardcode this
SCHEMA_VERSION = '1.0.0'


def datatable_to_metadata(datatable):
    dt_metadata = [
        {
            'name': col.name,
            'nullable': bool(col.to_pandas().isnull().any()),  # --> not sure this is what we mean by nullable
            'ordinal': datatable.to_pandas().columns.get_loc(col.name),
            'logical_type': {
                'parameters': _get_ltype_params(col.logical_type),
                'type': str(_get_ltype_class(col.logical_type))
            },
            'physical_type': {
                'parameters': {},  # --> col.to_pandas().dtype.__dict__ works for some but not all
                'type': str(col.dtype)
            },
            'semantic_tags': sorted(list(col.semantic_tags))
        }
        for col in datatable.columns.values()
    ]

    return {
        'name': datatable.name,
        'featuretools_version': FEATURETOOLS_VERSION,
        'woodwork_version': __version__,
        'schema_version': SCHEMA_VERSION,
        'metadata': dt_metadata
    }


def write_table_metadata(datatable, path, profile_name=None, **kwargs):
    if _is_s3(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, 'data'))
            dump_table_metadata(datatable, tmpdir, **kwargs)
            file_path = create_archive(tmpdir)

            transport_params = get_transport_params(profile_name)
            use_smartopen_es(file_path, path, read=False, transport_params=transport_params)
    elif _is_url(path):
        raise ValueError("Writing to URLs is not supported")
    else:
        path = os.path.abspath(path)
        dump_table_metadata(datatable, path)


def dump_table_metadata(datatable, path, **kwargs):
    file = os.path.join(path, 'table_metadata.json')

    with open(file, 'w') as file:
        json.dump(datatable_to_metadata(datatable), file)


def create_archive(tmpdir):
    file_name = "es-{date:%Y-%m-%d_%H%M%S}.tar".format(date=datetime.datetime.now())
    file_path = os.path.join(tmpdir, file_name)
    tar = tarfile.open(str(file_path), 'w')
    tar.add(str(tmpdir) + '/data_description.json', arcname='/data_description.json')
    tar.add(str(tmpdir) + '/data', arcname='/data')
    tar.close()
    return file_path
