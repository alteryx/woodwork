import json
import os

import pyarrow as pa

from woodwork.deserializers.deserializer_base import Deserializer, _check_schema_version
from woodwork.utils import _is_s3, _is_url


class ParquetDeserializer(Deserializer):
    """Deserialize Woodwork table from serialized data in parquet format."""

    format = "parquet"

    def deserialize(self, profile_name, validate):
        if _is_url(self.path) or _is_s3(self.path):
            dataframe = self.read_from_s3(profile_name)
        else:
            if self.data_subdirectory:
                self.path = os.path.join(self.path, self.data_subdirectory)
            self.read_path = os.path.join(self.path, self.filename)
            dataframe = self.read_from_local_path()
        dataframe.ww.init(**self.ww_init_dict, validate=validate)
        return dataframe

    def configure_deserializer(self):
        file_metadata = pa.parquet.read_metadata(self.read_path)
        self.typing_info = json.loads(file_metadata.metadata[b"ww_meta"])
        _check_schema_version(self.typing_info["schema_version"])
        loading_info = self.typing_info["loading_info"]
        self.kwargs = loading_info.get("params", {})
        self._set_init_dict(loading_info)

    def read_from_local_path(self):
        self.configure_deserializer()
        lib = self._get_library()
        return lib.read_parquet(self.read_path, engine=self.kwargs["engine"])
