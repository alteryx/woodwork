from woodwork.deserializers.deserializer_base import Deserializer
from woodwork.utils import import_or_none


class CSVDeserializer(Deserializer):
    """Deserialize Woodwork table from serialized data in CSV format."""

    format = "csv"

    def read_from_local_path(self):
        lib = self._get_library()
        return lib.read_csv(self.read_path, dtype=self.column_dtypes, **self.kwargs)

    def _use_pyarrow_engine(self):
        """Adds pyarrow engine to kwargs to improve performance in pandas."""
        if (
            "pandas" == self.typing_info["loading_info"]["table_type"]
            and bool(import_or_none("pyarrow.parquet"))
            and self.kwargs["engine"] == "python"
        ):
            self.kwargs.update(engine="pyarrow")

    def configure_deserializer(self):
        super().configure_deserializer()
        self._use_pyarrow_engine()
