from typing import Dict, Hashable, Union

import pandas as pd

from woodwork.utils import import_or_none

dd = import_or_none('dask.dataframe')
ks = import_or_none('databricks.koalas')

ColumnName = Hashable
UseStandardTagsDict = Dict[ColumnName, bool]
AnyDataFrame = pd.DataFrame
if dd:
    AnyDataFrame = Union[AnyDataFrame, dd.DataFrame]
if ks:
    AnyDataFrame = Union[AnyDataFrame, ks.DataFrame]
