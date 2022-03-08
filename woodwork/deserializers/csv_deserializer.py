from woodwork.deserializers.deserializer_base import Deserializer


class CSVDeserializer(Deserializer):
    format = "csv"

    def read_from_local_path(self):
        lib = self._get_library()
        return lib.read_csv(self.read_path, dtype=self.column_dtypes, **self.kwargs)
