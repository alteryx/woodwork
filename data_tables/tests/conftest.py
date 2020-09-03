import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='session')
def sample_df():
    return pd.DataFrame({
        'id': range(3),
        'full_name': ['Mr. John Doe', 'Doe, Mrs. Jane', 'James Brown'],
        'email': ['john.smith@example.com', np.nan, 'team@featuretools.com'],
        'phone_number': ['5555555555', '555-555-5555', '1-(555)-555-5555'],
        'age': [33, 25, 33],
        'signup_date': [pd.to_datetime('2020-09-01')] * 3,
        'is_registered': [True, False, True],
    })


@pytest.fixture(scope='session')
def sample_series():
    return pd.Series(['a', 'b', 'c'], name='sample_series').astype('object')
