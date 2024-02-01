import json
import os
import warnings
from pathlib import Path

import pandas as pd

from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.exceptions import ParametersIgnoredWarning, WoodworkFileExistsError
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
        if self.filename is not None and _is_dask_dataframe(dataframe):
            raise ValueError(
                "Writing a Dask dataframe to parquet with a filename specified is not supported",
            )
        if self.filename is not None and _is_spark_dataframe(dataframe):
            raise ValueError(
                "Writing a Spark dataframe to parquet with a filename specified is not supported",
            )
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
        Woodwork typing info before saving. Skip for Dask/Spark because for those formats
        typing information has to be added after files are saved to disk."""
        import pyarrow as pa

        if isinstance(self.dataframe, pd.DataFrame):
            dataframe = clean_latlong(self.dataframe)
            self.table = pa.Table.from_pandas(dataframe)

    def _generate_parquet_metadata(self):
        """Generate metadata for the parquet file header. For pandas this includes additional
        information needed by pandas. For Dask/Spark, this includes only the Woodwork typing info.
        """
        loading_info = {
            "location": self.location,
            "type": self.format,
            "params": self.kwargs,
        }
        self.typing_info["loading_info"].update(loading_info)
        # For Dask and Spark we only get the WW metadata because we haven't created
        # the pyarrow table yet, but for pandas we combine the existing parquet
        # metadata with the WW metadata.
        if _is_dask_dataframe(self.dataframe) or _is_spark_dataframe(self.dataframe):
            metadata = {
                "ww_meta".encode(): json.dumps(self.typing_info).encode(),
            }
        else:
            table_metadata = self.table.schema.metadata
            metadata = {
                "ww_meta".encode(): json.dumps(self.typing_info).encode(),
                **table_metadata,
            }
        self.metadata = metadata

    def _save_parquet_table_to_disk(self):
        """Writes data to disk with the updated metadata including WW typing info."""
        from pyarrow import parquet as pq

        if _is_dask_dataframe(self.dataframe):
            path, dataframe = self._setup_for_dask_and_spark()
            dataframe.to_parquet(path, custom_metadata=self.metadata)
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
                **self.metadata,
                **table_metadata,
            }
            table = table.replace_schema_metadata(combined_meta)
            pq.write_table(table, update_file, use_deprecated_int96_timestamps=True)

            # Remove checksum files which prevent deserialization if present due to updated parquet header
            crc_files = [f for f in files if Path(f).suffix == ".crc"]
            for file in crc_files:
                os.remove(os.path.join(path, file))
        else:
            file = self._get_filename()
            self.table = self.table.replace_schema_metadata(self.metadata)
            pq.write_table(self.table, file)

    def _setup_for_dask_and_spark(self):
        """Perform additional path setup required for Dask/Spark. Since Dask/Spark deserialize to
        directories only, the `_get_filename` method does not work like it does for pandas.
        """
        path = self.path
        if self.data_subdirectory is not None:
            path = os.path.join(path, self.data_subdirectory)
        if any([Path(f).suffix == ".parquet" for f in os.listdir(path)]):
            message = f"Data file already exists at '{path}'. "
            message += "Please remove or use a different directory."
            raise WoodworkFileExistsError(message)
        return path, clean_latlong(self.dataframe)
