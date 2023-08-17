<p align="center"><img width=50% src="https://alteryx-oss-web-images.s3.amazonaws.com/woodwork.svg" alt="Woodwork" /></p>
<p align="center">
    <a href="https://github.com/alteryx/woodwork/actions/workflows/tests_with_latest_deps.yaml" alt="Tests" target="_blank">
        <img src="https://github.com/alteryx/woodwork/actions/workflows/tests_with_latest_deps.yaml/badge.svg?branch=main" alt="Tests" />
    </a>
    <a href="https://codecov.io/gh/alteryx/woodwork">
        <img src="https://codecov.io/gh/alteryx/woodwork/branch/main/graph/badge.svg?token=KJCKMREBDP"/>
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
        <img src="https://static.pepy.tech/badge/woodwork/month" alt="PyPI Downloads" />
    </a>
</p>
<hr>

Woodwork provides a common typing namespace for using your existing DataFrames in Featuretools, EvalML, and general ML. A Woodwork
DataFrame stores the physical, logical, and semantic data types present in the data. In addition, it can store metadata about the data, allowing you to store specific information you might need for your application.

## Installation

Install with pip:

```bash
python -m pip install woodwork
```

or from the conda-forge channel on [conda](https://anaconda.org/conda-forge/woodwork):

```bash
conda install -c conda-forge woodwork
```

### Add-ons
**Update checker** - Receive automatic notifications of new Woodwork releases
```bash
python -m pip install "woodwork[updater]"
```

## Example

Below is an example of using Woodwork. In this example, a sample dataset of order items is used to create a Woodwork `DataFrame`, specifying the `LogicalType` for five of the columns.

```python
import pandas as pd
import woodwork as ww

df = pd.read_csv("https://oss.alteryx.com/datasets/online-retail-logs-2018-08-28.csv")
df.ww.init(name='retail')
df.ww.set_types(logical_types={
    'quantity': 'Integer',
    'customer_name': 'PersonFullName',
    'country': 'Categorical',
    'order_id': 'Categorical',
    'description': 'NaturalLanguage',
})
df.ww
```

```
                   Physical Type     Logical Type Semantic Tag(s)
Column
order_id                category      Categorical    ['category']
product_id              category      Categorical    ['category']
description               string  NaturalLanguage              []
quantity                   int64          Integer     ['numeric']
order_date        datetime64[ns]         Datetime              []
unit_price               float64           Double     ['numeric']
customer_name             string   PersonFullName              []
country                 category      Categorical    ['category']
total                    float64           Double     ['numeric']
cancelled                   bool          Boolean              []
```

We now have initialized Woodwork on the DataFrame with the specified logical types assigned. For columns that did not have a specified logical type value, Woodwork has automatically inferred the logical type based on the underlying data. Additionally, Woodwork has automatically assigned semantic tags to some of the columns, based on the inferred or assigned logical type.

If we wanted to do further analysis on only the columns in this table that have a logical type of `Boolean` or a semantic tag of `numeric` we can simply select those columns and access a dataframe containing just those columns:

```python
filtered_df = df.ww.select(include=['Boolean', 'numeric'])
filtered_df
```

```
    quantity  unit_price   total  cancelled
0          6      4.2075  25.245      False
1          6      5.5935  33.561      False
2          8      4.5375  36.300      False
3          6      5.5935  33.561      False
4          6      5.5935  33.561      False
..       ...         ...     ...        ...
95         6      4.2075  25.245      False
96       120      0.6930  83.160      False
97        24      0.9075  21.780      False
98        24      0.9075  21.780      False
99        24      0.9075  21.780      False
```

As you can see, Woodwork makes it easy to manage typing information for your data, and provides simple interfaces to access only the data you need based on the logical types or semantic tags. Please refer to the [Woodwork documentation](https://woodwork.alteryx.com/) for more detail on working with a Woodwork DataFrame.

## Support
The Woodwork community is happy to provide support to users of Woodwork. Project support can be found in four places depending on the type of question:
1. For usage questions, use [Stack Overflow](https://stackoverflow.com/questions/tagged/woodwork) with the `woodwork` tag.
2. For bugs, issues, or feature requests start a [Github issue](https://github.com/alteryx/woodwork/issues).
3. For discussion regarding development on the core library, use [Slack](https://join.slack.com/t/alteryx-oss/shared_invite/zt-182tyvuxv-NzIn6eiCEf8TBziuKp0bNA).
4. For everything else, the core developers can be reached by email at open_source_support@alteryx.com

## Built at Alteryx

**Woodwork** is an open source project built by [Alteryx](https://www.alteryx.com). To see the other open source projects we’re working on visit [Alteryx Open Source](https://www.alteryx.com/open-source). If building impactful data science pipelines is important to you or your business, please get in touch.

<p align="center">
  <a href="https://www.alteryx.com/open-source">
    <img src="https://alteryx-oss-web-images.s3.amazonaws.com/OpenSource_Logo-01.png" alt="Alteryx Open Source" width="800"/>
  </a>
</p>
