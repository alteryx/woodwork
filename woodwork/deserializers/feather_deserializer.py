import pandas as pd

from woodwork.deserializers.deserializer_base import Deserializer


class FeatherDeserializer(Deserializer):
    """Deserialize Woodwork table from serialized data in feather format."""

    format = "feather"

    def read_from_local_path(self):
        return pd.read_feather(self.read_path)
