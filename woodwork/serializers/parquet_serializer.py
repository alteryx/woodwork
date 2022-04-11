import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.exceptions import WoodworkFileExistsError
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
        super().__init__(path, filename, data_subdirectory, typing_info_filename)
        self.typing_info_filename = None

    def serialize(self, dataframe, profile_name, **kwargs):
        import_or_raise("pyarrow", PYARROW_IMPORT_ERROR_MESSAGE)
        if self.filename is not None and _is_dask_dataframe(dataframe):
            raise ValueError(
                "Writing a Dask dataframe to parquet with a filename specified is not supported"
            )
        if self.filename is not None and _is_spark_dataframe(dataframe):
            raise ValueError(
                "Writing a Spark dataframe to parquet with a filename specified is not supported"
            )
        self.kwargs["engine"] = "pyarrow"
        return super().serialize(dataframe, profile_name, **kwargs)

    def write_dataframe(self):
        if _is_dask_dataframe(self.dataframe) or _is_spark_dataframe(self.dataframe):
            pass
        else:
            dataframe = clean_latlong(self.dataframe)
            self.table = pa.Table.from_pandas(dataframe)

    def write_typing_info(self):
        loading_info = {
            "location": self.location,
            "type": self.format,
            "params": self.kwargs,
        }
        self.typing_info["loading_info"].update(loading_info)
        if _is_dask_dataframe(self.dataframe):
            combined_meta = {
                "ww_meta".encode(): json.dumps(self.typing_info).encode(),
            }
        elif _is_spark_dataframe(self.dataframe):
            combined_meta = {}
        else:
            table_metadata = self.table.schema.metadata
            combined_meta = {
                "ww_meta".encode(): json.dumps(self.typing_info).encode(),
                **table_metadata,
            }
        self._save_parquet_table_to_disk(combined_meta)

    def _save_parquet_table_to_disk(self, metadata):
        if _is_dask_dataframe(self.dataframe):
            path, dataframe = self._setup_for_dask_and_spark()
            dataframe.to_parquet(path, custom_metadata=metadata)
        elif _is_spark_dataframe(self.dataframe):
            path, dataframe = self._setup_for_dask_and_spark()
            dataframe.to_parquet(path)
            files = os.listdir(path)

            # Update first parquet file to save WW metadata
            parquet_files = sorted([f for f in files if Path(f).suffix == ".parquet"])
            update_file = os.path.join(path, parquet_files[0])
            table = pq.read_table(update_file)
            table_metadata = table.schema.metadata
            combined_meta = {
                "ww_meta".encode(): json.dumps(self.typing_info).encode(),
                **table_metadata,
            }
            table = table.replace_schema_metadata(combined_meta)
            pq.write_table(table, update_file)

            # Remove checksum files which prevent deserialization if present due to updated parquet header
            crc_files = [f for f in files if Path(f).suffix == ".crc"]
            for file in crc_files:
                os.remove(os.path.join(path, file))
        else:
            file = self._get_filename()
            self.table = self.table.replace_schema_metadata(metadata)
            pq.write_table(self.table, file)

    def _setup_for_dask_and_spark(self):
        path = self.path
        if self.data_subdirectory is not None:
            path = os.path.join(path, self.data_subdirectory)
        if any([Path(f).suffix == ".parquet" for f in os.listdir(path)]):
            message = f"Data file already exists at '{path}'. "
            message += "Please remove or use a different directory."
            raise WoodworkFileExistsError(message)
        return path, clean_latlong(self.dataframe)
