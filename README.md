<p align="center"><img width=50% src="https://raw.githubusercontent.com/FeatureLabs/woodwork/main/docs/source/images/woodwork_light.png" alt="Woodwork" /></p>

# Woodwork

[![codecov](https://codecov.io/gh/FeatureLabs/woodwork/branch/main/graph/badge.svg?token=KJCKMREBDP)](https://codecov.io/gh/FeatureLabs/woodwork)
[![PyPI version](https://badge.fury.io/py/woodwork.svg?maxAge=2592000)](https://badge.fury.io/py/woodwork)
[![Downloads](https://pepy.tech/badge/woodwork/month)](https://pepy.tech/project/woodwork/month)

Woodwork provides you with a common DataTable object to use with Featuretools, EvalML, and general ML. A DataTable object contains the physical, logical, and semantic data types present in the data. In addition, it can store metadata about the data.

## Installation

Install with pip:

```bash
python -m pip install woodwork
```

## Example

Below is an example of using Woodwork.

```python
import woodwork as ww

data = ww.demo.load_retail(nrows=100)
     
dt = ww.DataTable(data, name="retail")

dt.set_logical_types({
    'quantity': 'Double',
    'customer_name': 'Categorical',
    'country': 'Categorical'
})

dt.types
```

## Built at Alteryx Innovation Labs

<a href="https://www.alteryx.com/innovation-labs">
    <img src="https://evalml-web-images.s3.amazonaws.com/alteryx_innovation_labs.png" alt="Alteryx Innovation Labs" />
</a>
