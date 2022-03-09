import pandas as pd

from woodwork.serializers.serializer_base import Serializer


class PickleSerializer(Serializer):
    """Serialize a Woodwork table to disk as a pickle file."""

    format = "pickle"

    def write_dataframe(self):
        if not isinstance(self.dataframe, pd.DataFrame):
            msg = "DataFrame type not compatible with pickle serialization. Please serialize to another format."
            raise ValueError(msg)

        file = self._get_filename()
        self.dataframe.to_pickle(file, **self.kwargs)
