import sys
import warnings

from .type_system import type_system

if sys.version_info.major == 3 and sys.version_info.minor == 7:  # pragma: no cover
    warnings.warn(
        "Woodwork may not support Python 3.7 in next non-bugfix release.", FutureWarning
    )
