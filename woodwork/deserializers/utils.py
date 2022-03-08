import json
import os
import tarfile
import tempfile
from pathlib import Path

from woodwork.deserializers import (
    ArrowDeserializer,
    CSVDeserializer,
    FeatherDeserializer,
    OrcDeserializer,
    ParquetDeserializer,
    PickleDeserializer,
)
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.utils import _is_s3, _is_url

FORMAT_TO_DESERIALIZER = {
    CSVDeserializer.format: CSVDeserializer,
    PickleDeserializer.format: PickleDeserializer,
    ParquetDeserializer.format: ParquetDeserializer,
    ArrowDeserializer.format: ArrowDeserializer,
    FeatherDeserializer.format: FeatherDeserializer,
    OrcDeserializer.format: OrcDeserializer,
}


def get_deserializer(
    path, filename, data_subdirectory, typing_info_filename, profile_name
):
    """Determine the proper Deserializer class to use based on the specified parameters.
    Initializes and returns the proper Deserializer object."""
    typing_info = read_table_typing_information(
        path, typing_info_filename, profile_name
    )
    format = typing_info["loading_info"]["type"]

    deserializer_cls = FORMAT_TO_DESERIALIZER.get(format)

    return deserializer_cls(path, filename, data_subdirectory, typing_info)


def read_table_typing_information(path, typing_info_filename, profile_name):
    """Read Woodwork typing information from disk, S3 path, or URL.

    Args:
        path (str): Location on disk, S3 path, or URL to read typing info file.
        typing_info_filename (str): Name of JSON file in which typing info is stored.
        profile_name (str, bool): The AWS profile specified to access to S3.

    Returns:
        dict: Woodwork typing information dictionary
    """
    if _is_url(path) or _is_s3(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = Path(path).name
            file_path = os.path.join(tmpdir, file_name)
            transport_params = None

            if _is_s3(path):
                transport_params = get_transport_params(profile_name)

            use_smartopen(file_path, path, transport_params)
            with tarfile.open(str(file_path)) as tar:
                tar.extractall(path=tmpdir)

            file = os.path.join(tmpdir, typing_info_filename)
            with open(file, "r") as file:
                typing_info = json.load(file)
    else:
        path = os.path.abspath(path)
        assert os.path.exists(path), '"{}" does not exist'.format(path)
        file = os.path.join(path, typing_info_filename)
        with open(file, "r") as file:
            typing_info = json.load(file)
        typing_info["path"] = path

    return typing_info
