# flake8: noqa
import pkg_resources
import sys
import warnings

from .config import config
from .type_sys import type_system
from .type_sys.utils import list_logical_types, list_semantic_tags
from .utils import concat_columns, read_file
from .version import __version__

import woodwork.column_accessor
import woodwork.demo
import woodwork.table_accessor
from woodwork.accessor_utils import (
    get_invalid_schema_message,
    init_series,
    is_schema_valid,
)

if sys.version_info.major == 3 and sys.version_info.minor == 7:  # pragma: no cover
    warnings.warn(
        "Woodwork may not support Python 3.7 in next non-bugfix release.", FutureWarning
    )

# Call functions registered by other libraries when woodwork is imported
for entry_point in pkg_resources.iter_entry_points(
    "alteryx_open_src_initialize"
):  # pragma: no cover
    try:
        method = entry_point.load()
        if callable(method):
            method("woodwork")
    except Exception:
        pass
