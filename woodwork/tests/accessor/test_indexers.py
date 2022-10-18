import pandas as pd
import pytest

from woodwork.accessor_utils import _is_dask_dataframe, _is_dask_series
from woodwork.exceptions import WoodworkNotInitError
from woodwork.indexers import _iLocIndexer, _locIndexer
from woodwork.logical_types import (
    Categorical,
    Datetime,
    Double,
    EmailAddress,
    Integer,
    PhoneNumber,
)
from woodwork.tests.testing_utils import to_pandas
from woodwork.utils import import_or_none

dd = import_or_none("dask.dataframe")


def test_iLocIndexer_class_error(sample_df_dask, sample_series_dask):
    with pytest.raises(TypeError, match="iloc is not supported for Dask DataFrames"):
        _iLocIndexer(sample_df_dask)

    with pytest.raises(TypeError, match="iloc is not supported for Dask Series"):
        _iLocIndexer(sample_series_dask)


def test_iLocIndexer_class(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail("iloc is not supported with Dask inputs")
    sample_df.ww.init()
    ind = _iLocIndexer(sample_df)
    pd.testing.assert_frame_equal(to_pandas(ind.data), to_pandas(sample_df))
    pd.testing.assert_frame_equal(to_pandas(ind[1:2]), to_pandas(sample_df.iloc[1:2]))
    assert ind[0, 0] == 0


def test_locIndexer_class(sample_df):
    sample_df.ww.init()
    ind = _locIndexer(sample_df)
    pd.testing.assert_frame_equal(to_pandas(ind.data), to_pandas(sample_df))
    pd.testing.assert_frame_equal(to_pandas(ind[1:2]), to_pandas(sample_df.loc[1:2]))
    single_val = ind[0, "id"]
    if _is_dask_series(single_val):
        # Dask returns a series - convert to pandas to check the value
        single_val = single_val.compute()
        assert len(single_val) == 1
        single_val = single_val.loc[0]
    assert single_val == 0


def test_error_before_table_init(sample_df):
    error_message = "Woodwork not initialized for this DataFrame. Initialize by calling DataFrame.ww.init"

    with pytest.raises(WoodworkNotInitError, match=error_message):
        sample_df.ww.iloc[:, :]

    with pytest.raises(WoodworkNotInitError, match=error_message):
        sample_df.ww.loc[:, :]


def test_error_before_column_init(sample_series):
    error_message = (
        "Woodwork not initialized for this Series. Initialize by calling Series.ww.init"
    )

    with pytest.raises(WoodworkNotInitError, match=error_message):
        sample_series.ww.iloc[:]

    with pytest.raises(WoodworkNotInitError, match=error_message):
        sample_series.ww.loc[:]


def test_iloc_column(sample_series):
    if _is_dask_series(sample_series):
        pytest.xfail("iloc is not supported with Dask inputs")
    series = sample_series.copy()
    logical_type = Categorical
    semantic_tags = ["tag1", "tag2"]
    description = "custom column description"
    origin = "base"
    metadata = {"meta_key": "custom metadata"}
    series.ww.init(
        logical_type=logical_type,
        semantic_tags=semantic_tags,
        description=description,
        origin=origin,
        metadata=metadata,
    )

    sliced = series.ww.iloc[2:]
    assert sliced.name == "sample_series"
    assert isinstance(sliced.ww.logical_type, logical_type)
    assert sliced.ww.semantic_tags == {"category", "tag1", "tag2"}
    assert sliced.ww.description == description
    assert sliced.ww.origin == origin
    assert sliced.ww.metadata == metadata
    pd.testing.assert_series_equal(to_pandas(sliced), to_pandas(series.iloc[2:]))

    assert series.ww.iloc[0] == "a"

    series = sample_series.copy()
    series.ww.init(use_standard_tags=False)
    sliced = series.ww.iloc[:]
    assert sliced.name
    assert isinstance(sliced.ww.logical_type, logical_type)
    assert sliced.ww.semantic_tags == set()


def test_iloc_column_does_not_propagate_changes_to_data(sample_series):
    if _is_dask_series(sample_series):
        pytest.xfail("iloc is not supported with Dask inputs")
    logical_type = Categorical
    semantic_tags = ["tag1", "tag2"]
    description = "custom column description"
    origin = "base"
    metadata = {"meta_key": "custom metadata"}
    sample_series.ww.init(
        logical_type=logical_type,
        semantic_tags=semantic_tags,
        description=description,
        origin=origin,
        metadata=metadata,
        use_standard_tags=False,
    )

    sliced = sample_series.ww.iloc[2:]
    sample_series.ww.add_semantic_tags("new_tag")
    assert sliced.ww.semantic_tags == {"tag1", "tag2"}
    assert sliced.ww.semantic_tags is not sample_series.ww.semantic_tags

    sample_series.ww.metadata["new_key"] = "new_value"
    assert sliced.ww.metadata == {"meta_key": "custom metadata"}
    assert sliced.ww.metadata is not sample_series.ww.metadata


def test_loc_column(sample_series):
    series = sample_series.copy()
    logical_type = Categorical
    semantic_tags = ["tag1", "tag2"]
    series.ww.init(logical_type=logical_type, semantic_tags=semantic_tags)

    sliced = series.ww.loc[2:]
    assert sliced.name == "sample_series"
    assert isinstance(sliced.ww.logical_type, logical_type)
    assert sliced.ww.semantic_tags == {"category", "tag1", "tag2"}
    pd.testing.assert_series_equal(to_pandas(sliced), to_pandas(series.loc[2:]))

    single_val = series.ww.loc[0]

    if _is_dask_series(series):
        # Dask returns a series - convert to pandas to check the value
        single_val = single_val.compute()
        assert len(single_val) == 1
        single_val = single_val.loc[0]
    assert single_val == "a"

    series = sample_series.copy()
    series.ww.init(use_standard_tags=False)
    sliced = series.ww.loc[:]
    assert sliced.name
    assert isinstance(sliced.ww.logical_type, logical_type)
    assert sliced.ww.semantic_tags == set()


def test_iloc_indices_column(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail("iloc is not supported with Dask inputs")
    sample_df.ww.init(index="id", time_index="signup_date")
    sliced_index = sample_df.ww.iloc[:, 0]
    assert sliced_index.ww.semantic_tags == {"index"}

    sliced_time_index = sample_df.ww.iloc[:, 5]
    assert sliced_time_index.ww.semantic_tags == {"time_index"}


def test_loc_indices_column(sample_df):
    sample_df.ww.init(index="id", time_index="signup_date")
    sliced_index = sample_df.ww.loc[:, "id"]
    assert sliced_index.ww.semantic_tags == {"index"}

    sliced_time_index = sample_df.ww.loc[:, "signup_date"]
    assert sliced_time_index.ww.semantic_tags == {"time_index"}


def test_indexer_uses_standard_tags(sample_df):
    sample_df.ww.init(
        index="id",
        time_index="age",
        use_standard_tags={"id": False, "age": True},
    )
    sliced_index = sample_df.ww.loc[:, "id"]
    assert sliced_index.ww.semantic_tags == {"index"}

    sliced_time_index = sample_df.ww.loc[:, "age"]
    assert sliced_time_index.ww.semantic_tags == {"numeric", "time_index"}


def test_iloc_with_properties(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail("iloc is not supported with Dask inputs")
    semantic_tags = {
        "full_name": "tag1",
        "email": ["tag2"],
        "phone_number": ["tag3", "tag2"],
        "signup_date": {"secondary_time_index"},
    }
    logical_types = {
        "full_name": Categorical,
        "email": EmailAddress,
        "phone_number": PhoneNumber,
        "age": Double,
    }
    df = sample_df.copy()
    df.ww.init(logical_types=logical_types, semantic_tags=semantic_tags)
    sliced = df.ww.iloc[1:3, 1:3]
    assert sliced.shape == (2, 2)
    assert sliced.ww.semantic_tags == {
        "full_name": {"category", "tag1"},
        "email": {"tag2"},
    }
    assert isinstance(sliced.ww.logical_types["full_name"], Categorical)
    assert isinstance(sliced.ww.logical_types["email"], EmailAddress)
    assert sliced.ww.index is None

    df = sample_df.copy()
    df.ww.init(logical_types=logical_types, use_standard_tags=False)
    sliced = df.ww.iloc[:, [0, 5]]
    assert sliced.ww.semantic_tags == {"id": set(), "signup_date": set()}
    assert isinstance(sliced.ww.logical_types["id"], Integer)
    assert isinstance(sliced.ww.logical_types["signup_date"], Datetime)
    assert sliced.ww.index is None


def test_iloc_dimensionality(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail("iloc is not supported with Dask inputs")
    semantic_tags = {
        "full_name": "tag1",
        "email": ["tag2"],
        "phone_number": ["tag3", "tag2"],
        "signup_date": {"secondary_time_index"},
    }
    logical_types = {
        "full_name": Categorical,
        "email": EmailAddress,
        "phone_number": PhoneNumber,
        "age": Double,
    }
    sample_df.ww.init(logical_types=logical_types, semantic_tags=semantic_tags)

    sliced_series_row = sample_df.ww.iloc[1]
    assert isinstance(sliced_series_row, pd.Series)
    assert set(sliced_series_row.index) == set(sample_df.columns)
    assert sliced_series_row.name == 1

    sliced_series_col = sample_df.ww.iloc[:, 1]
    assert isinstance(sliced_series_col.ww.logical_type, Categorical)
    assert sliced_series_col.ww.semantic_tags == {"tag1", "category"}
    assert sliced_series_col.ww.name == "full_name"


def test_iloc_indices(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail("iloc is not supported with Dask inputs")
    df_with_index = sample_df.copy()
    df_with_index.ww.init(index="id")
    assert df_with_index.ww.iloc[:, [0, 5]].ww.index == "id"
    assert df_with_index.ww.iloc[:, [1, 2]].ww.index is None

    df_with_time_index = sample_df.copy()
    df_with_time_index.ww.init(time_index="signup_date")
    assert df_with_time_index.ww.iloc[:, [0, 5]].ww.time_index == "signup_date"
    assert df_with_time_index.ww.iloc[:, [1, 2]].ww.index is None


def test_iloc_table_does_not_propagate_changes_to_data(sample_df):
    if _is_dask_dataframe(sample_df):
        pytest.xfail("iloc is not supported with Dask inputs")
    sample_df.ww.init()
    sliced = sample_df.ww.iloc[1:3, 1:3]

    sample_df.ww.add_semantic_tags({"full_name": "new_tag"})
    assert sliced.ww.semantic_tags["full_name"] == set()
    assert (
        sliced.ww.semantic_tags["full_name"]
        is not sample_df.ww.semantic_tags["full_name"]
    )

    sample_df.ww.metadata["new_key"] = "new_value"
    assert sliced.ww.metadata == {}
    assert sliced.ww.metadata is not sample_df.ww.metadata

    sample_df.ww.columns["email"].metadata["new_key"] = "new_value"
    assert sliced.ww.columns["email"].metadata == {}
    assert (
        sliced.ww.columns["email"].metadata
        is not sample_df.ww.columns["email"].metadata
    )
