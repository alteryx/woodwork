import urllib.request

import pytest

from woodwork import DataTable
from woodwork.demo import load_retail, load_retail_to_accessor
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage
)
from woodwork.schema import Schema


@pytest.fixture(autouse=True)
def set_testing_headers():
    opener = urllib.request.build_opener()
    opener.addheaders = [('Testing', 'True')]
    urllib.request.install_opener(opener)


def test_load_retail_diff():
    nrows = 10
    df = load_retail(nrows=nrows, return_dataframe=True)
    assert df.shape[0] == nrows
    nrows_second = 11
    df = load_retail(nrows=nrows_second, return_dataframe=True)
    assert df.shape[0] == nrows_second

    assert 'order_product_id' in df.columns
    assert df['order_product_id'].is_unique


def test_load_retail_datatable():
    dt = load_retail(nrows=10, return_dataframe=False)
    assert isinstance(dt, DataTable)

    expected_logical_types = {
        'order_product_id': Categorical,
        'order_id': Categorical,
        'product_id': Categorical,
        'description': NaturalLanguage,
        'quantity': Integer,
        'order_date': Datetime,
        'unit_price': Double,
        'customer_name': Categorical,
        'country': Categorical,
        'total': Double,
        'cancelled': Boolean,
    }

    for column in dt.columns.values():
        assert column.logical_type == expected_logical_types[column.name]

    assert dt.index == 'order_product_id'
    assert dt.time_index == 'order_date'
    assert dt.columns['order_product_id'].semantic_tags == {'index'}
    assert dt.columns['order_date'].semantic_tags == {'time_index'}


def test_load_retail_to_accessor_diff():
    nrows = 10
    df = load_retail_to_accessor(nrows=nrows, init_woodwork=False)
    assert df.ww.schema is None
    assert df.shape[0] == nrows
    nrows_second = 11
    df = load_retail_to_accessor(nrows=nrows_second, init_woodwork=False)
    assert df.ww.schema is None
    assert df.shape[0] == nrows_second

    assert 'order_product_id' in df.columns
    assert df['order_product_id'].is_unique


def test_load_retail_to_accessor():
    df = load_retail_to_accessor(nrows=10, init_woodwork=True)
    assert isinstance(df.ww.schema, Schema)

    expected_logical_types = {
        'order_product_id': Categorical,
        'order_id': Categorical,
        'product_id': Categorical,
        'description': NaturalLanguage,
        'quantity': Integer,
        'order_date': Datetime,
        'unit_price': Double,
        'customer_name': Categorical,
        'country': Categorical,
        'total': Double,
        'cancelled': Boolean,
    }

    for col_name, column in df.ww.columns.items():
        assert column['logical_type'] == expected_logical_types[col_name]

    assert df.ww.index == 'order_product_id'
    assert df.ww.time_index == 'order_date'
    assert df.ww.semantic_tags['order_product_id'] == {'index'}
    assert df.ww.semantic_tags['order_date'] == {'time_index'}
