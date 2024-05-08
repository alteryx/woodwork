import glob
import os

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
        self.kwargs = {**self.default_kwargs, **kwargs}
        return super().serialize(dataframe, profile_name, **kwargs)

    def _get_filename(self):
        if self.filename is None:
            ww_name = self.dataframe.ww.name or "data"
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
        dataframe = self.dataframe
        file = self._get_filename()
        dataframe.to_csv(file, **csv_kwargs)
