# DataTables

DataTable are common data objects to use with Featuretools, EvalML, and general ML. A DataTable object contains the physical, logical, and semantic data types present in the data. In addition, it stores metadata about the data.


## Installation
Clone repo

```bash
git clone https://github.com/FeatureLabs/data_tables.git
```

Install with pip

```
cd data_tables
python -m pip install data_tables/
```

or from the Conda-forge channel on [conda](https://anaconda.org/conda-forge/featuretools):

```
conda install -c conda-forge featuretools
```

### Add-ons

You can install add-ons individually or all at once by running

```
python -m pip install featuretools[complete]
```

**Update checker** - Receive automatic notifications of new Featuretools releases

```
python -m pip install featuretools[update_checker]
```

**TSFresh Primitives** - Use 60+ primitives from [tsfresh](https://tsfresh.readthedocs.io/en/latest/) within Featuretools

```
python -m pip install featuretools[tsfresh]
```

## Example
Below is an example of using Deep Feature Synthesis (DFS) to perform automated feature engineering. In this example, we apply DFS to a multi-table dataset consisting of timestamped customer transactions.

```python
>> import featuretools as ft
>> es = ft.demo.load_mock_customer(return_entityset=True)
>> es.plot()
```

## Built at Alteryx Innovation Labs
<a href="https://www.alteryx.com/innovation-labs">
    <img src="https://evalml-web-images.s3.amazonaws.com/alteryx_innovation_labs.png" alt="Alteryx Innovation Labs" />
</a>
