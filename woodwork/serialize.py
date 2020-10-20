import json
import os

from woodwork.utils import _get_ltype_class, _get_ltype_params
from .version import __version__

FEATURETOOLS_VERSION = '0.20.0'  # shouldnt have to hardcode this
SCHEMA_VERSION = '1.0.0'


def datatable_to_metadata(datatable):
    dt_metadata = [
        {'name': col_name,
            'nullable': bool(col.to_pandas().isnull().any()),  # --> not sure this is what we mean by nullable
            'ordinal': datatable.to_pandas().columns.get_loc(col_name),
            'logical_type': {
                'parameters': _get_ltype_params(col.logical_type),
                'type': str(_get_ltype_class(col.logical_type))
            },
            'physical_type': {
                'parameters': {},  # --> col.to_pandas().dtype.__dict__ works for some but not all
                'type': str(col.dtype)
            },
            'tags': sorted(list(col.semantic_tags))}
        for col_name, col in datatable.columns.items()
    ]

    return {
        'name': datatable.name,
        'featuretools_version': FEATURETOOLS_VERSION,
        'woodwork_version': __version__,
        'schema_version': SCHEMA_VERSION,
        'metadata': dt_metadata
    }


def write_data(datatable, path):
    # get working locally and then add s3
    path = os.path.abspath(path)
    dump_table_metadata(datatable, path)


def dump_table_metadata(datatable, path):
    metadata = datatable_to_metadata(datatable)
    file = os.path.join(path, 'table_metadata.json')

    with open(file, 'w') as file:
        json.dump(metadata, file)
