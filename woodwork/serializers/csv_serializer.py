import glob
import os

from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.exceptions import WoodworkFileExistsError
from woodwork.serializers.serializer_base import Serializer


class CSVSerializer(Serializer):
    """Serialize a Woodwork table to disk as a CSV file."""

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
        if _is_spark_dataframe(dataframe):
            if self.filename is not None:
                raise ValueError(
                    "Writing a Spark dataframe to csv with a filename specified is not supported",
                )
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
        self.location = basename
        if self.data_subdirectory:
            self.location = os.path.join(self.data_subdirectory, basename)
        location = os.path.join(self.write_path, self.location)
        if os.path.exists(location) or glob.glob(location):
            message = f"Data file already exists at '{location}'."
            message += "Please remove or use a different filename."
            raise WoodworkFileExistsError(message)
        return location

    def write_dataframe(self):
        csv_kwargs = self.kwargs.copy()
        # engine kwarg not needed for writing, only reading
        if "engine" in csv_kwargs.keys():
            del csv_kwargs["engine"]
        if _is_spark_dataframe(self.dataframe):
            dataframe = self.dataframe.ww.copy()
            columns = list(dataframe.select_dtypes("object").columns)
            dataframe[columns] = dataframe[columns].astype(str)
            csv_kwargs["compression"] = str(csv_kwargs["compression"])
        else:
            dataframe = self.dataframe
        file = self._get_filename()
        dataframe.to_csv(file, **csv_kwargs)
