import json
import os

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
        if _is_dask_dataframe(self.dataframe):
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
        else:
            table_metadata = self.table.schema.metadata
            combined_meta = {
                "ww_meta".encode(): json.dumps(self.typing_info).encode(),
                **table_metadata,
            }
        self._save_parquet_table_to_disk(combined_meta)

    def _save_parquet_table_to_disk(self, metadata):
        if _is_dask_dataframe(self.dataframe):
            dataframe = clean_latlong(self.dataframe)
            path = self.path
            if self.data_subdirectory is not None:
                path = os.path.join(path, self.data_subdirectory)
            if os.path.exists(os.path.join(path, "part.0.parquet")):
                message = f"Data file already exists at '{path}'. "
                message += "Please remove or use a different directory."
                raise WoodworkFileExistsError(message)
            dataframe.to_parquet(path, custom_metadata=metadata)
        else:
            file = self._get_filename()
            self.table = self.table.replace_schema_metadata(metadata)
            pq.write_table(self.table, file)
