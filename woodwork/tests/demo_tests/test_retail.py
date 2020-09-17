import urllib.request

import pytest

from woodwork.demo import load_retail


@pytest.fixture(autouse=True)
def set_testing_headers():
    opener = urllib.request.build_opener()
    opener.addheaders = [('Testing', 'True')]
    urllib.request.install_opener(opener)


def test_load_retail_diff():
    nrows = 10
    df = load_retail(nrows=nrows)
    assert df.shape[0] == nrows
    nrows_second = 11
    df = load_retail(nrows=nrows_second)
    assert df.shape[0] == nrows_second
