from woodwork.serializers.serializer_base import Serializer


class PickleSerializer(Serializer):
    """Serialize a Woodwork table to disk as a pickle file."""

    format = "pickle"

    def write_dataframe(self):
        file = self._get_filename()
        self.dataframe.to_pickle(file, **self.kwargs)
