def validate_subset_dt(subset_dt, dt):
    assert subset_dt.name == dt.name
    assert len(subset_dt.columns) == len(subset_dt.to_pandas().columns)
    for subset_col_name, subset_col in subset_dt.columns.items():
        assert subset_col_name in dt.columns
        col = dt.columns[subset_col_name]
        assert subset_col.logical_type == col.logical_type
        assert subset_col.semantic_tags == col.semantic_tags
        assert subset_col.dtype == col.dtype
        assert subset_col.to_pandas().equals(col.to_pandas())


def mi_between_cols(col1, col2, df):
    mi_series = df.loc[df['column_1'] == col1].loc[df['column_2'] == col2]['mutual_info']

    if len(mi_series) == 0:
        mi_series = df.loc[df['column_1'] == col2].loc[df['column_2'] == col1]['mutual_info']

    return mi_series.iloc[0]
