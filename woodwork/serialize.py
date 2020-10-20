import json
import os

from woodwork.utils import _get_ltype_class, _get_ltype_args
from .version import __version__

FEATURETOOLS_VERSION = '0.20.0'  # shouldnt have to hardcode this
SCHEMA_VERSION = '0.1.0'


def datatable_to_metadata(datatable):
    dt_metadata = [
        {'name': col_name,
            'nullable': col.to_pandas().isnull().any(),  # --> not sure this is what we mean by nullable
            'ordinal': datatable.to_pandas().columns.get_loc(col_name),
            'logical_type': {
                'parameters': _get_ltype_args(col.logical_type),
                'type': _get_ltype_class(col.logical_type)
            },
            'physical_type': {
                'parameters': {},  # --> col.to_pandas().dtype.__dict__ works for some but not all
                'type': col.dtype
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
