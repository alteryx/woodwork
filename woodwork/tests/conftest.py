import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(params=['sample_df_pandas', 'sample_df_dask'])
def sample_df(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_df_pandas():
    return pd.DataFrame({
        'id': range(3),
        'full_name': ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown'],
        'email': ['john.smith@example.com', np.nan, 'team@featuretools.com'],
        'phone_number': ['5555555555', '555-555-5555', '1-(555)-555-5555'],
        'age': [33, 25, 33],
        'signup_date': [pd.to_datetime('2020-09-01')] * 3,
        'is_registered': [True, False, True],
    })


@pytest.fixture()
def sample_df_dask(sample_df_pandas):
    return dd.from_pandas(sample_df_pandas, npartitions=2)


@pytest.fixture(params=['sample_series_pandas', 'sample_series_dask'])
def sample_series(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_series_pandas():
    return pd.Series(['a', 'b', 'c', 'a', np.nan], name='sample_series').astype('object')


@pytest.fixture()
def sample_series_dask(sample_series_pandas):
    return dd.from_pandas(sample_series_pandas, npartitions=2)


@pytest.fixture(params=['sample_datetime_series_pandas', 'sample_datetime_series_dask'])
def sample_datetime_series(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def sample_datetime_series_pandas():
    return pd.Series([pd.to_datetime('2020-09-01')] * 4, name='sample_datetime_series').astype('object')


@pytest.fixture()
def sample_datetime_series_dask(sample_datetime_series_pandas):
    return dd.from_pandas(sample_datetime_series_pandas, npartitions=2)
