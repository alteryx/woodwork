from timeit import default_timer as timer

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_integer_dtype
from sklearn.metrics.cluster import normalized_mutual_info_score

from woodwork.accessor_utils import _is_dask_dataframe, _is_koalas_dataframe
from woodwork.logical_types import Datetime, Double, LatLong, Timedelta
from woodwork.utils import (
    _is_latlong_nan,
    _update_progress,
    get_valid_mi_types,
    import_or_none,
)

dd = import_or_none("dask.dataframe")
ks = import_or_none("databricks.koalas")
