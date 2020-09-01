import pandas as pd
import pytest

from data_tables import DataTable


def test_datatable_init(sample_df):
    dt = DataTable(sample_df)
    df = dt.df

    assert dt.name is None
    assert dt.index is None
    assert dt.time_index is None
    assert isinstance(df, pd.DataFrame)
    assert set(dt.columns.keys()) == set(sample_df.columns)
    assert df is not sample_df
    pd.testing.assert_frame_equal(df, sample_df)


def test_datatable_init_with_name_and_index_vals(sample_df):
    dt = DataTable(sample_df,
                   name='datatable',
                   index='id',
                   time_index='signup_date')

    assert dt.name == 'datatable'
    assert dt.index == 'id'
    assert dt.time_index == 'signup_date'


def test_raises_error_with_invalid_dataframe_type():
    error_message = 'Dataframe must be a pandas.DataFrame'
    with pytest.raises(AssertionError, match=error_message):
        DataTable(pd.Series())


def test_raises_error_with_invalid_name_type(sample_df):
    error_message = 'DataTable name must be a string'
    with pytest.raises(AssertionError, match=error_message):
        DataTable(sample_df, name=1)
