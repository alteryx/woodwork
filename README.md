<p align="center"><img width=50% src="https://woodwork-web-images.s3.amazonaws.com/woodwork.svg" alt="Woodwork" /></p>
<p align="center">
    <a href="https://github.com/alteryx/woodwork/actions?query=branch%3Amain+workflow%3ATests" target="_blank">
        <img src="https://github.com/alteryx/woodwork/workflows/Tests/badge.svg?branch=main" alt="Tests" />
    </a>
    <a href="https://woodwork.alteryx.com/en/latest/?badge=stable" target="_blank">
        <img src="https://readthedocs.com/projects/feature-labs-inc-datatables/badge/?version=stable" alt="Documentation Status" />
    </a>
    <a href="https://badge.fury.io/py/woodwork" target="_blank">
        <img src="https://badge.fury.io/py/woodwork.svg?maxAge=2592000" alt="PyPI Version" />
    </a>
    <a href="https://anaconda.org/conda-forge/woodwork" target="_blank">
        <img src="https://anaconda.org/conda-forge/woodwork/badges/version.svg" alt="Anaconda Version" />
    </a>
    <a href="https://pepy.tech/project/woodwork" target="_blank">
        <img src="https://pepy.tech/badge/woodwork/month" alt="PyPI Downloads" />
    </a>
</p>
<hr>

Woodwork provides you with a common DataTable object to use with Featuretools, EvalML, and general ML. A DataTable object contains the physical, logical, and semantic data types present in the data. In addition, it can store metadata about the data.

## Installation

Install with pip:

```bash
python -m pip install woodwork
```

or from the conda-forge channel on [conda](https://anaconda.org/conda-forge/woodwork):

```bash
conda install -c conda-forge woodwork
```

## Example

Below is an example of using Woodwork. In this example, a sample dataset of order items is used to create a Woodwork `DataTable`, specifying the `LogicalType` for three of the columns.

```python
import woodwork as ww

data = ww.demo.load_retail(nrows=100, return_dataframe=True)
dt = ww.DataTable(data, name='retail')
dt.set_types(logical_types={
    'quantity': 'Double',
    'customer_name': 'Categorical',
    'country': 'Categorical'
})
dt
```

```
                Physical Type     Logical Type Semantic Tag(s)
Data Column
order_id                Int64          Integer       [numeric]
product_id           category      Categorical      [category]
description            string  NaturalLanguage              []
quantity              float64           Double       [numeric]
order_date     datetime64[ns]         Datetime              []
unit_price            float64           Double       [numeric]
customer_name        category      Categorical      [category]
country              category      Categorical      [category]
total                 float64           Double       [numeric]
```

We now have created a Woodwork `DataTable` with the specified logical types assigned. For columns that did not have a specified logical type value, Woodwork has automatically inferred the logical type based on the underlying data. Additionally, Woodwork has automatically assigned semantic tags to some of the columns, based on the inferred or assigned logical type.

If we wanted to do further analysis on only the columns in this table that have a logical type of `Boolean` or a semantic tag of `numeric` we can simply select those columns and access a dataframe containing just those columns:

```python
filtered_df = dt.select(include=['Boolean', 'numeric']).to_dataframe()
filtered_df
```

```
    order_id  quantity  unit_price   total  cancelled
0     536365       6.0      4.2075  25.245      False
1     536365       6.0      5.5935  33.561      False
2     536365       8.0      4.5375  36.300      False
3     536365       6.0      5.5935  33.561      False
4     536365       6.0      5.5935  33.561      False
..       ...       ...         ...     ...        ...
95    536378       6.0      4.2075  25.245      False
96    536378     120.0      0.6930  83.160      False
97    536378      24.0      0.9075  21.780      False
98    536378      24.0      0.9075  21.780      False
99    536378      24.0      0.9075  21.780      False
```

As you can see, Woodwork makes it easy to manage typing information for your data, and provides simple interfaces to access only the data you need based on the logical types or semantic tags. Please refer to the Woodwork documentation for more detail on working with Woodwork tables.

## Built at Alteryx Innovation Labs

<a href="https://www.alteryx.com/innovation-labs">
    <img src="https://evalml-web-images.s3.amazonaws.com/alteryx_innovation_labs.png" alt="Alteryx Innovation Labs" />
</a>
