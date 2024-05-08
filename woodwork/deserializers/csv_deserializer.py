import pandas as pd

from woodwork.deserializers.deserializer_base import Deserializer


class CSVDeserializer(Deserializer):
    """Deserialize Woodwork table from serialized data in CSV format."""

    format = "csv"

    def read_from_local_path(self):
        return pd.read_csv(self.read_path, dtype=self.column_dtypes, **self.kwargs)
