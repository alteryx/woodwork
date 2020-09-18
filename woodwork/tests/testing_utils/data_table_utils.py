def validate_subset_dt(subset_dt, dt):
    assert subset_dt.name == dt.name
    assert len(subset_dt.columns) == len(subset_dt.df.columns)
    for subset_col_name, subset_col in subset_dt.columns.items():
        assert subset_col_name in dt.columns
        col = dt.columns[subset_col_name]
        assert subset_col.logical_type == col.logical_type
        assert subset_col.semantic_tags == col.semantic_tags
        assert subset_col.dtype == col.dtype
        assert subset_col.series.equals(col.series)
