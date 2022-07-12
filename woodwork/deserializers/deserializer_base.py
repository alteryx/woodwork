import os
import tarfile
import tempfile
import warnings
from itertools import zip_longest
from pathlib import Path

import pandas as pd

import woodwork as ww
from woodwork.exceptions import OutdatedSchemaWarning, UpgradeSchemaWarning
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.serializers.serializer_base import SCHEMA_VERSION
from woodwork.utils import _is_s3, _is_url, import_or_raise

PYARROW_IMPORT_ERROR_MESSAGE_DESERIALIZE = (
    f"The pyarrow library is required to deserialize from {format}.\n"
    "Install via pip:\n"
    "    pip install pyarrow\n"
    "Install via conda:\n"
    "   conda install pyarrow -c conda-forge"
)


class Deserializer:
    def __init__(self, path, filename, data_subdirectory, typing_info):
        self.path = path
        self.filename = filename
        self.data_subdirectory = data_subdirectory
        self.typing_info = typing_info

    def deserialize(self, profile_name, validate):
        """Reconstruct Woodwork dataframe from saved data and typing information"""
        self.configure_deserializer()
        if _is_url(self.path) or _is_s3(self.path):
            dataframe = self.read_from_s3(profile_name)
        else:
            dataframe = self.read_from_local_path()
        dataframe.ww.init(**self.ww_init_dict, validate=validate)
        return dataframe

    def configure_deserializer(self):
        """Extract info from typing information required to read data and initialize Woodwork"""
        _check_schema_version(self.typing_info["schema_version"])
        loading_info = self.typing_info["loading_info"]
        if not (_is_s3(self.path) or _is_url(self.path)):
            path = self.typing_info["path"]
            self.read_path = os.path.join(path, loading_info["location"])
        self.kwargs = loading_info.get("params", {})
        self._set_init_dict(loading_info)

    def _set_init_dict(self, loading_info):
        logical_types = {}
        semantic_tags = {}
        column_descriptions = {}
        column_origins = {}
        column_metadata = {}
        use_standard_tags = {}
        self.column_dtypes = {}
        for col in self.typing_info["column_typing_info"]:
            col_name = col["name"]

            ltype_metadata = col["logical_type"]
            ltype = ww.type_system.str_to_logical_type(
                ltype_metadata["type"],
                params=ltype_metadata["parameters"],
            )

            tags = col["semantic_tags"]

            if "index" in tags:
                tags.remove("index")
            elif "time_index" in tags:
                tags.remove("time_index")

            logical_types[col_name] = ltype
            semantic_tags[col_name] = tags
            column_descriptions[col_name] = col["description"]
            column_origins[col_name] = col["origin"]
            column_metadata[col_name] = col["metadata"]
            use_standard_tags[col_name] = col["use_standard_tags"]

            col_type = col["physical_type"]["type"]
            table_type = loading_info.get("table_type", "pandas")
            if col_type == "category":
                # Make sure categories are recreated properly
                cat_values = col["physical_type"]["cat_values"]
                cat_dtype = col["physical_type"]["cat_dtype"]
                if table_type == "pandas":
                    cat_object = pd.CategoricalDtype(
                        pd.Index(cat_values, dtype=cat_dtype),
                    )
                else:
                    cat_object = pd.CategoricalDtype(pd.Series(cat_values))
                col_type = cat_object
            elif table_type == "spark" and col_type == "object":
                col_type = "string"
            self.column_dtypes[col_name] = col_type

        if "index" in self.kwargs.keys():
            del self.kwargs["index"]

        self.ww_init_dict = {
            "name": self.typing_info.get("name"),
            "index": self.typing_info.get("index"),
            "time_index": self.typing_info.get("time_index"),
            "logical_types": logical_types,
            "semantic_tags": semantic_tags,
            "use_standard_tags": use_standard_tags,
            "table_metadata": self.typing_info.get("table_metadata"),
            "column_metadata": column_metadata,
            "column_descriptions": column_descriptions,
            "column_origins": column_origins,
        }

    def read_from_s3(self, profile_name):
        """Read data from S3 into a dataframe"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_filename = Path(self.path).name
            tar_filepath = os.path.join(tmpdir, tar_filename)
            transport_params = None

            if _is_s3(self.path):
                transport_params = get_transport_params(profile_name)

            use_smartopen(tar_filepath, self.path, transport_params)
            with tarfile.open(str(tar_filepath)) as tar:
                tar.extractall(path=tmpdir)
            self.read_path = os.path.join(
                tmpdir,
                self.typing_info["loading_info"]["location"],
            )
            return self.read_from_local_path()

    def read_from_local_path(self):
        """Read data from a local location into a dataframe"""
        raise NotImplementedError(
            "Must define read_from_local_path on Deserializer subclass",
        )  # pragma: no cover

    def _get_library(self):
        table_type = self.typing_info["loading_info"]["table_type"]
        if table_type == "dask":
            DASK_ERR_MSG = (
                "Cannot load Dask DataFrame - unable to import Dask.\n\n"
                "Please install with pip or conda:\n\n"
                'python -m pip install "woodwork[dask]"\n\n'
                "conda install dask"
            )
            lib = import_or_raise("dask.dataframe", DASK_ERR_MSG)
        elif table_type == "spark":
            SPARK_ERR_MSG = (
                "Cannot load Spark DataFrame - unable to import Spark.\n\n"
                "Please install with pip or conda:\n\n"
                'python -m pip install "woodwork[spark]"\n\n'
                "conda install spark\n\n"
                "conda install pyspark"
            )
            lib = import_or_raise("pyspark.pandas", SPARK_ERR_MSG)
            if "compression" in self.kwargs.keys():
                self.kwargs["compression"] = str(self.kwargs["compression"])
        else:
            lib = pd

        return lib


def _check_schema_version(saved_version_str):
    """Warns users if the schema used to save their data is greater than the latest
    supported schema or if it is an outdated schema that is no longer supported."""
    saved = saved_version_str.split(".")
    current = SCHEMA_VERSION.split(".")

    for c_num, s_num in zip_longest(current, saved, fillvalue=0):
        if int(c_num) > int(s_num):
            break
        elif int(c_num) < int(s_num):
            warnings.warn(
                UpgradeSchemaWarning().get_warning_message(
                    saved_version_str,
                    SCHEMA_VERSION,
                ),
                UpgradeSchemaWarning,
            )
            break

    # Check if saved has older major version.
    if int(current[0]) > int(saved[0]):
        warnings.warn(
            OutdatedSchemaWarning().get_warning_message(saved_version_str),
            OutdatedSchemaWarning,
        )
