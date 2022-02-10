from .constants import CANDIDATE_COLUMN_NAME


def _build_freq_dataframe(df):
    def to_series(row):
        for alias in row[CANDIDATE_COLUMN_NAME]:
            row[alias] = True
        return row

    df = df.apply(to_series, axis=1)
    df = df.fillna(False)

    return df
