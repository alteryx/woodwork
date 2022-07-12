import datetime
import json
import os
import tarfile
import tempfile

from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.exceptions import WoodworkFileExistsError
from woodwork.logical_types import LatLong
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.type_sys.utils import _get_ltype_class, _get_specified_ltype_params
from woodwork.utils import _is_s3, _is_url

SCHEMA_VERSION = "12.0.0"

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
        """Serialize data and typing information to disk."""
        self.dataframe = dataframe
        self.typing_info = typing_info_to_dict(self.dataframe)

        if _is_s3(self.path):
            self.save_to_s3(profile_name)
        elif _is_url(self.path):
            raise ValueError("Writing to URLs is not supported")
        else:
            self.write_path = os.path.abspath(self.path)
            self.save_to_local_path()

    def save_to_local_path(self):
        """Serialize data and typing information to a local directory."""
        if self.data_subdirectory:
            location = os.path.join(self.write_path, self.data_subdirectory)
            os.makedirs(location, exist_ok=True)
        else:
            os.makedirs(self.write_path, exist_ok=True)
        self.write_dataframe()
        self.write_typing_info()

    def save_to_s3(self, profile_name):
        """Serialize data and typing information to S3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_path = tmpdir
            self.save_to_local_path()
            archive_file_path = self._create_archive()
            transport_params = get_transport_params(profile_name)
            use_smartopen(
                archive_file_path,
                self.path,
                read=False,
                transport_params=transport_params,
            )

    def write_dataframe(self):
        """Save dataframe to disk."""
        raise NotImplementedError(
            "Must define write_dataframe on Serializer subclass",
        )  # pragma: no cover

    def write_typing_info(self):
        """Save Woodwork typing information JSON file to disk."""
        loading_info = {
            "location": self.location,
            "type": self.format,
            "params": self.kwargs,
        }
        self.typing_info["loading_info"].update(loading_info)
        file = os.path.join(self.write_path, self.typing_info_filename)

        if os.path.exists(file):
            message = f"Typing info already exists at '{file}'. "
            message += "Please remove or use a different filename."
            raise WoodworkFileExistsError(message)

        try:
            with open(file, "w") as file:
                json.dump(self.typing_info, file)
        except TypeError:
            raise TypeError(
                "Woodwork table is not json serializable. Check table and column metadata for values that may not be serializable.",
            )

    def _get_filename(self):
        """Get the full filepath that should be used to save the data."""
        if self.filename is None:
            ww_name = self.dataframe.ww.name or "data"
            basename = ".".join([ww_name, self.format])
        else:
            basename = self.filename
        self.location = basename
        if self.data_subdirectory:
            self.location = os.path.join(self.data_subdirectory, basename)
        location = os.path.join(self.write_path, self.location)
        if os.path.exists(location):
            message = f"Data file already exists at '{location}'. "
            message += "Please remove or use a different filename."
            raise WoodworkFileExistsError(message)
        return location

    def _create_archive(self):
        """Create a tar archive of data and typing information."""
        file_name = "ww-{date:%Y-%m-%d_%H%M%S}.tar".format(date=datetime.datetime.now())
        file_path = os.path.join(self.write_path, file_name)
        tar = tarfile.open(str(file_path), "w")
        if self.typing_info_filename:
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
    elif _is_spark_dataframe(dataframe):
        table_type = "spark"
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


def clean_latlong(dataframe):
    """Convert latlong tuples to strings for parquet, arrow and feather file format.
    Attempting to serialize with tuples present results in an error"""
    latlong_columns = [
        col_name
        for col_name, col in dataframe.ww.columns.items()
        if _get_ltype_class(col.logical_type) == LatLong
    ]
    if len(latlong_columns) > 0:
        dataframe = dataframe.ww.copy()
        dataframe[latlong_columns] = dataframe[latlong_columns].astype(str)

    return dataframe
