# DataTables

[![Coverage Status](https://codecov.io/gh/FeatureLabs/datatables/branch/master/graph/badge.svg)](https://codecov.io/gh/FeatureLabs/datatables)

DataTable are common data objects to use with Featuretools, EvalML, and general ML. A DataTable object contains the physical, logical, and semantic data types present in the data. In addition, it stores metadata about the data.

## Installation

Clone repo

```bash
git clone https://github.com/FeatureLabs/datatables.git
cd datatables
```

Install with pip in editable mode

```bash
python -m pip install -e .
```

## Example

Below is an example of using DataTables.

```python
from data_tables import DataTable
from data_tables.logical_types import Datetime, Categorical, NaturalLanguage

import pandas as pd

df = pd.read_csv(...)

dt = DataTable(df,
               name='retail', # default to df.name
               index=None,
               time_index=None)

dt.set_types({
    "datetime": Datetime,
    "comments": NaturalLanguage,
    "store_id": Categorical
})
```

## Built at Alteryx Innovation Labs

<a href="https://www.alteryx.com/innovation-labs">
    <img src="https://evalml-web-images.s3.amazonaws.com/alteryx_innovation_labs.png" alt="Alteryx Innovation Labs" />
</a>
