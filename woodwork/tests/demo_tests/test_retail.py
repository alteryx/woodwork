import urllib.request

import pytest

from woodwork import DataTable
from woodwork.demo import load_retail
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage
)


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
