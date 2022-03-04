import datetime
import json
import os
import tarfile
import tempfile

from woodwork.accessor_utils import _is_dask_dataframe, _is_koalas_dataframe
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.serialize import typing_info_to_dict
from woodwork.utils import _is_s3, _is_url


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
        raise NotImplementedError

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
        super().serialize(dataframe, profile_name, **kwargs)

    def write_dataframe(self):
        if self.filename is None:
            ww_name = self.dataframe.ww.name or "data"
            if _is_dask_dataframe(self.dataframe):
                basename = "{}-*.{}".format(ww_name, self.format)
            else:
                basename = ".".join([ww_name, self.format])
        else:
            basename = self.filename
        self.location = os.path.join(self.data_subdirectory, basename)
        file = os.path.join(self.write_path, self.location)

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
        dataframe.to_csv(file, **csv_kwargs)

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


class ParquetSerializer(Serializer):
    format = "parquet"


class ArrowSerializer(Serializer):
    format = "arrow"


class FeatherSerializer(Serializer):
    format = "feather"


class OrcSerializer(Serializer):
    format = "orc"


class PickleSerializer(Serializer):
    format = "pickle"
