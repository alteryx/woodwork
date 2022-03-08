from woodwork.deserializers.deserializer_base import Deserializer


class OrcDeserializer(Deserializer):
    """Deserialize Woodwork table from serialized data in orc format."""

    format = "orc"

    def read_from_local_path(self):
        lib = self._get_library()
        return lib.read_orc(self.read_path)
