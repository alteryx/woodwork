import datetime
import json
import os
import tarfile
import tempfile
from mimetypes import add_type, guess_type

import pandas as pd

from woodwork.accessor_utils import _is_dask_dataframe, _is_koalas_dataframe
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.serializer_utils import clean_latlong
from woodwork.type_sys.utils import _get_ltype_class, _get_specified_ltype_params
from woodwork.utils import _is_s3, _is_url, import_or_raise

SCHEMA_VERSION = "11.3.0"
FORMATS = ["csv", "pickle", "parquet", "arrow", "feather", "orc"]
PYARROW_IMPORT_ERROR_MESSAGE = (
    f"The pyarrow library is required to serialize to {format}.\n"
    "Install via pip:\n"
    "    pip install pyarrow\n"
    "Install via conda:\n"
    "   conda install pyarrow -c conda-forge"
)


class Serializer:
    def __init__(self, path, filename, data_subdirectory, typing_info_filename):
        self.path = path
        self.write_path = None
        self.filename = filename
        self.data_subdirectory = data_subdirectory
        self.typing_info_filename = typing_info_filename
        self.dataframe = None
        self.typing_info = None
        self.location = None
        self.kwargs = {}

    def serialize(self, dataframe, profile_name, **kwargs):
        self.dataframe = dataframe
        self.profile_name = profile_name
        self.typing_info = typing_info_to_dict(self.dataframe)

        if _is_s3(self.path):
            self.save_to_s3()
        elif _is_url(self.path):
            raise ValueError("Writing to URLs is not supported")
        else:
            self.write_path = os.path.abspath(self.path)
            self.save_to_local_path()

    def save_to_local_path(self):
        os.makedirs(
            os.path.join(self.write_path, self.data_subdirectory), exist_ok=True
        )
        self.write_dataframe()
        self.write_typing_info()

    def save_to_s3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_path = tmpdir
            self.save_to_local_path()
            archive_file_path = self._create_archive()
            transport_params = get_transport_params(self.profile_name)
            use_smartopen(
                archive_file_path,
                self.path,
                read=False,
                transport_params=transport_params,
            )

    def write_dataframe(self):
        raise NotImplementedError

    def write_typing_info(self):
        loading_info = {
            "location": self.location,
            "type": self.format,
            "params": self.kwargs,
        }
        self.typing_info["loading_info"].update(loading_info)
        try:
            file = os.path.join(self.write_path, self.typing_info_filename)
            with open(file, "w") as file:
                json.dump(self.typing_info, file)
        except TypeError:
            raise TypeError(
                "Woodwork table is not json serializable. Check table and column metadata for values that may not be serializable."
            )

    def _get_filename(self):
        if self.filename is None:
            ww_name = self.dataframe.ww.name or "data"
            basename = ".".join([ww_name, self.format])
        else:
            basename = self.filename
        self.location = os.path.join(self.data_subdirectory, basename)
        return os.path.join(self.write_path, self.location)

    def _create_archive(self):
        file_name = "ww-{date:%Y-%m-%d_%H%M%S}.tar".format(date=datetime.datetime.now())
        file_path = os.path.join(self.write_path, file_name)
        tar = tarfile.open(str(file_path), "w")
        tar.add(
            str(self.write_path) + f"/{self.typing_info_filename}",
            arcname=f"/{self.typing_info_filename}",
        )
        tar.add(
            str(self.write_path) + f"/{self.data_subdirectory}",
            arcname=f"/{self.data_subdirectory}",
        )
        tar.close()
        return file_path


class CSVSerializer(Serializer):
    format = "csv"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_kwargs = {
            "sep": ",",
            "encoding": "utf-8",
            "engine": "python",
            "index": False,
        }

    def serialize(self, dataframe, profile_name, **kwargs):
        if _is_koalas_dataframe(dataframe):
            self.default_kwargs["multiline"] = True
            self.default_kwargs["ignoreLeadingWhitespace"] = False
            self.default_kwargs["ignoreTrailingWhitespace"] = False
        self.kwargs = {**self.default_kwargs, **kwargs}
        return super().serialize(dataframe, profile_name, **kwargs)

    def _get_filename(self):
        if self.filename is None:
            ww_name = self.dataframe.ww.name or "data"
            if _is_dask_dataframe(self.dataframe):
                basename = "{}-*.{}".format(ww_name, self.format)
            else:
                basename = ".".join([ww_name, self.format])
        else:
            basename = self.filename
        self.location = os.path.join(self.data_subdirectory, basename)
        return os.path.join(self.write_path, self.location)

    def write_dataframe(self):
        csv_kwargs = self.kwargs.copy()
        # engine kwarg not needed for writing, only reading
        if "engine" in csv_kwargs.keys():
            del csv_kwargs["engine"]
        if _is_koalas_dataframe(self.dataframe):
            dataframe = self.dataframe.ww.copy()
            columns = list(dataframe.select_dtypes("object").columns)
            dataframe[columns] = dataframe[columns].astype(str)
            csv_kwargs["compression"] = str(csv_kwargs["compression"])
        else:
            dataframe = self.dataframe
        file = self._get_filename()
        dataframe.to_csv(file, **csv_kwargs)


class ParquetSerializer(Serializer):
    format = "parquet"

    def serialize(self, dataframe, profile_name, **kwargs):
        import_or_raise("pyarrow", PYARROW_IMPORT_ERROR_MESSAGE)
        self.kwargs["engine"] = "pyarrow"
        return super().serialize(dataframe, profile_name, **kwargs)

    def write_dataframe(self):
        file = self._get_filename()
        dataframe = clean_latlong(self.dataframe)
        dataframe.to_parquet(file, **self.kwargs)


class FeatherSerializer(Serializer):
    format = "feather"

    def serialize(self, dataframe, profile_name, **kwargs):
        import_or_raise("pyarrow", PYARROW_IMPORT_ERROR_MESSAGE)
        return super().serialize(dataframe, profile_name, **kwargs)

    def write_dataframe(self):
        file = self._get_filename()
        dataframe = clean_latlong(self.dataframe)
        dataframe.to_feather(file, **self.kwargs)


class ArrowSerializer(FeatherSerializer):
    format = "arrow"


class OrcSerializer(Serializer):
    format = "orc"

    def serialize(self, dataframe, profile_name, **kwargs):
        import_or_raise("pyarrow", PYARROW_IMPORT_ERROR_MESSAGE)
        # Serialization to orc relies on pyarrow.Table.from_pandas which doesn't work with Dask
        if _is_dask_dataframe(dataframe):
            msg = "DataFrame type not compatible with orc serialization. Please serialize to another format."
            raise ValueError(msg)
        self.kwargs["engine"] = "pyarrow"
        return super().serialize(dataframe, profile_name, **kwargs)

    def write_dataframe(self):
        from pyarrow import Table, orc

        file = self._get_filename()
        dataframe = clean_latlong(self.dataframe)
        df = dataframe.copy()
        for c in df:
            if df[c].dtype.name == "category":
                df[c] = df[c].astype("string")
        pa_table = Table.from_pandas(df, preserve_index=False)
        orc.write_table(pa_table, file)


class PickleSerializer(Serializer):
    format = "pickle"

    def write_dataframe(self):
        if not isinstance(self.dataframe, pd.DataFrame):
            msg = "DataFrame type not compatible with pickle serialization. Please serialize to another format."
            raise ValueError(msg)

        file = self._get_filename()
        self.dataframe.to_pickle(file, **self.kwargs)


# Dictionary mapping content types to the appropriate format specifier
CONTENT_TYPE_TO_FORMAT = {
    "text/csv": "csv",
    "application/parquet": "parquet",
    "application/arrow": "arrow",
    "application/feather": "feather",
    "application/orc": "orc",
}

# Dictionary used to get the corret serializer from a given format
FORMAT_TO_SERIALIZER = {
    CSVSerializer.format: CSVSerializer,
    PickleSerializer.format: PickleSerializer,
    ParquetSerializer.format: ParquetSerializer,
    ArrowSerializer.format: ArrowSerializer,
    FeatherSerializer.format: FeatherSerializer,
    OrcSerializer.format: OrcSerializer,
}

# Add new mimetypes
add_type("application/parquet", ".parquet")
add_type("application/arrow", ".arrow")
add_type("application/feather", ".feather")
add_type("application/orc", ".orc")


def get_serializer(format=None, filename=None):
    """Get serializer class based on format or filename"""
    if format is None and filename is not None:
        content_type, _ = guess_type(filename)
        format = CONTENT_TYPE_TO_FORMAT.get(content_type)
        if format is None:
            raise RuntimeError(
                "Content type could not be inferred. Please specify content_type and try again."
            )

    serializer = FORMAT_TO_SERIALIZER.get(format)

    if serializer is None:
        error = "must be one of the following formats: {}"
        raise ValueError(error.format(", ".join(FORMAT_TO_SERIALIZER.keys())))

    return serializer


def typing_info_to_dict(dataframe):
    """Creates the description for a Woodwork table, including typing information for each column
    and loading information.

    Args:
        dataframe (pd.DataFrame, dd.Dataframe, ks.DataFrame): DataFrame with Woodwork typing
            information initialized.

    Returns:
        dict: Dictionary containing Woodwork typing information
    """
    if _is_dask_dataframe(dataframe):
        # Need to determine the category info for Dask it can be saved below
        category_cols = [
            colname
            for colname, col in dataframe.ww._schema.columns.items()
            if col.is_categorical
        ]
        dataframe = dataframe.ww.categorize(columns=category_cols)
    ordered_columns = dataframe.columns

    def _get_physical_type_dict(column):
        type_dict = {"type": str(column.dtype)}
        if str(column.dtype) == "category":
            type_dict["cat_values"] = column.dtype.categories.to_list()
            type_dict["cat_dtype"] = str(column.dtype.categories.dtype)
        return type_dict

    column_typing_info = [
        {
            "name": col_name,
            "ordinal": ordered_columns.get_loc(col_name),
            "use_standard_tags": col.use_standard_tags,
            "logical_type": {
                "parameters": _get_specified_ltype_params(col.logical_type),
                "type": str(_get_ltype_class(col.logical_type)),
            },
            "physical_type": _get_physical_type_dict(dataframe[col_name]),
            "semantic_tags": sorted(list(col.semantic_tags)),
            "description": col.description,
            "origin": col.origin,
            "metadata": col.metadata,
        }
        for col_name, col in dataframe.ww.columns.items()
    ]

    if _is_dask_dataframe(dataframe):
        table_type = "dask"
    elif _is_koalas_dataframe(dataframe):
        table_type = "koalas"
    else:
        table_type = "pandas"

    return {
        "schema_version": SCHEMA_VERSION,
        "name": dataframe.ww.name,
        "index": dataframe.ww.index,
        "time_index": dataframe.ww.time_index,
        "column_typing_info": column_typing_info,
        "loading_info": {"table_type": table_type},
        "table_metadata": dataframe.ww.metadata,
    }
