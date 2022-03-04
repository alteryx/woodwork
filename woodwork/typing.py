from typing import Dict, Hashable, Union

import pandas as pd

from woodwork.utils import import_or_none

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")

ColumnName = Hashable
UseStandardTagsDict = Dict[ColumnName, bool]
AnyDataFrame = pd.DataFrame
if dd:
    AnyDataFrame = Union[AnyDataFrame, dd.DataFrame]
if ps:
    AnyDataFrame = Union[AnyDataFrame, ps.DataFrame]
