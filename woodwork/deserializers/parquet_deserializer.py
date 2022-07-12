import json
import os
import tarfile
import tempfile
from pathlib import Path

from woodwork.deserializers.deserializer_base import (
    PYARROW_IMPORT_ERROR_MESSAGE_DESERIALIZE,
    Deserializer,
    _check_schema_version,
)
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.utils import _is_s3, _is_url, import_or_raise


class ParquetDeserializer(Deserializer):
    """Deserialize Woodwork table from serialized data in parquet format."""

    format = "parquet"

    def deserialize(self, profile_name, validate):
        import_or_raise("pyarrow", PYARROW_IMPORT_ERROR_MESSAGE_DESERIALIZE)
        if _is_url(self.path) or _is_s3(self.path):
            dataframe = self.read_from_s3(profile_name)
        else:
            if self.data_subdirectory:
                self.path = os.path.join(self.path, self.data_subdirectory)
            self.read_path = self.path
            if self.filename:
                self.read_path = os.path.join(self.path, self.filename)

            dataframe = self.read_from_local_path()
        dataframe.ww.init(**self.ww_init_dict, validate=validate)
        return dataframe

    def configure_deserializer(self):
        import pyarrow as pa

        self._set_metadata_path()
        file_metadata = pa.parquet.read_metadata(self.metadata_path)
        self.typing_info = json.loads(file_metadata.metadata[b"ww_meta"])
        _check_schema_version(self.typing_info["schema_version"])
        loading_info = self.typing_info["loading_info"]
        self.kwargs = loading_info.get("params", {})
        self._set_init_dict(loading_info)

    def read_from_local_path(self):
        self.configure_deserializer()
        lib = self._get_library()
        return lib.read_parquet(self.read_path, engine=self.kwargs["engine"])

    def read_from_s3(self, profile_name):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_filename = Path(self.path).name
            tar_filepath = os.path.join(tmpdir, tar_filename)
            transport_params = None

            if _is_s3(self.path):
                transport_params = get_transport_params(profile_name)

            use_smartopen(tar_filepath, self.path, transport_params)
            with tarfile.open(str(tar_filepath)) as tar:
                tar.extractall(path=tmpdir)

            self.read_path = os.path.join(tmpdir, self.data_subdirectory, self.filename)

            return self.read_from_local_path()

    def _set_metadata_path(self):
        # If we are reading a single pandas file, we get the metadata from the file.
        # If we are reading into Dask/Spark we need to get the metadata from the
        # first file that was serialized.
        self.metadata_path = self.read_path
        if os.path.isdir(self.read_path):
            files = os.listdir(self.read_path)
            if "part.0.parquet" in files:
                # Dask will serialize with "part.*.parquet" file names
                self.metadata_path = os.path.join(self.read_path, "part.0.parquet")
            elif any(["snappy.parquet" in f for f in files]):
                # Spark will serialize files with a unique hash but with the ".snappy.parquet" extension
                parquet_files = sorted(
                    [f for f in files if Path(f).suffix == ".parquet"],
                )
                self.metadata_path = os.path.join(self.read_path, parquet_files[0])
