import pandas as pd

from woodwork.deserializers.deserializer_base import Deserializer


class PickleDeserializer(Deserializer):
    """Deserialize Woodwork table from serialized data in pickle format."""

    format = "pickle"

    def read_from_local_path(self):
        return pd.read_pickle(self.read_path, **self.kwargs)
