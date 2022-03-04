import json
import os
import tarfile
import tempfile
from pathlib import Path

from woodwork.logical_types import LatLong
from woodwork.s3_utils import get_transport_params, use_smartopen
from woodwork.type_sys.utils import _get_ltype_class
from woodwork.utils import _is_s3, _is_url


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


def read_table_typing_information(path, typing_info_file, profile_name):
    """Read Woodwork typing information from disk, S3 path, or URL.

    Args:
        path (str): Location on disk, S3 path, or URL to read typing info file.
        typing_info_file (str): Name of JSON file in which typing info is stored.

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

            file = os.path.join(tmpdir, typing_info_file)
            with open(file, "r") as file:
                typing_info = json.load(file)
    else:
        path = os.path.abspath(path)
        assert os.path.exists(path), '"{}" does not exist'.format(path)
        file = os.path.join(path, typing_info_file)
        with open(file, "r") as file:
            typing_info = json.load(file)
        typing_info["path"] = path

    return typing_info


def save_orc_file(dataframe, filepath):
    from pyarrow import Table, orc

    df = dataframe.copy()
    for c in df:
        if df[c].dtype.name == "category":
            df[c] = df[c].astype("string")
    pa_table = Table.from_pandas(df, preserve_index=False)
    orc.write_table(pa_table, filepath)
