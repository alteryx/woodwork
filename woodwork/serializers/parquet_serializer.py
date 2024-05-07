import json
import os
import warnings

from woodwork.exceptions import ParametersIgnoredWarning
from woodwork.serializers.serializer_base import (
    PYARROW_IMPORT_ERROR_MESSAGE,
    Serializer,
    clean_latlong,
)
from woodwork.utils import import_or_raise


class ParquetSerializer(Serializer):
    """Serialize a Woodwork table to disk as a parquet file."""

    format = "parquet"

    def __init__(self, path, filename, data_subdirectory, typing_info_filename):
        import_or_raise("pyarrow", PYARROW_IMPORT_ERROR_MESSAGE)
        super().__init__(path, filename, data_subdirectory, typing_info_filename)
        if typing_info_filename and typing_info_filename != "woodwork_typing_info.json":
            warnings.warn(
                "Typing info filename has been ignored. Typing information will be stored in parquet file header.",
                ParametersIgnoredWarning,
            )
        self.typing_info_filename = None

    def serialize(self, dataframe, profile_name, **kwargs):
        self.kwargs["engine"] = "pyarrow"
        return super().serialize(dataframe, profile_name, **kwargs)

    def save_to_local_path(self):
        """Serialize data and typing information to a local directory. Overrides method on base class
        due to different serialiation flow"""
        if self.data_subdirectory:
            location = os.path.join(self.write_path, self.data_subdirectory)
            os.makedirs(location, exist_ok=True)
        else:
            os.makedirs(self.write_path, exist_ok=True)
        self._create_pyarrow_table()
        self._generate_parquet_metadata()
        self._save_parquet_table_to_disk()

    def _create_pyarrow_table(self):
        """Create a pyarrow table for pandas. This table will get updated to included
        Woodwork typing info before saving."""
        import pyarrow as pa

        dataframe = clean_latlong(self.dataframe)
        self.table = pa.Table.from_pandas(dataframe)

    def _generate_parquet_metadata(self):
        """Generate metadata for the parquet file header. For pandas this includes additional
        information needed by pandas.
        """
        loading_info = {
            "location": self.location,
            "type": self.format,
            "params": self.kwargs,
        }
        self.typing_info["loading_info"].update(loading_info)
        table_metadata = self.table.schema.metadata
        metadata = {
            "ww_meta".encode(): json.dumps(self.typing_info).encode(),
            **table_metadata,
        }
        self.metadata = metadata

    def _save_parquet_table_to_disk(self):
        """Writes data to disk with the updated metadata including WW typing info."""
        from pyarrow import parquet as pq

        file = self._get_filename()
        self.table = self.table.replace_schema_metadata(self.metadata)
        pq.write_table(self.table, file)
