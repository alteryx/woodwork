from woodwork.deserializers.deserializer_base import Deserializer


class ParquetDeserializer(Deserializer):
    """Deserialize Woodwork table from serialized data in parquet format."""

    format = "parquet"

    def read_from_local_path(self):
        lib = self._get_library()
        return lib.read_parquet(self.read_path, engine=self.kwargs["engine"])
