from woodwork.deserializers.deserializer_base import Deserializer


class FeatherDeserializer(Deserializer):
    """Deserialize Woodwork table from serialized data in feather format."""

    format = "feather"

    def read_from_local_path(self):
        lib = self._get_library()
        return lib.read_feather(self.read_path)
