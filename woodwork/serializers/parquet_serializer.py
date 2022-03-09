from woodwork.accessor_utils import _is_dask_dataframe, _is_spark_dataframe
from woodwork.serializers.serializer_base import (
    PYARROW_IMPORT_ERROR_MESSAGE,
    Serializer,
    clean_latlong,
)
from woodwork.utils import import_or_raise


class ParquetSerializer(Serializer):
    """Serialize a Woodwork table to disk as a parquet file."""

    format = "parquet"

    def serialize(self, dataframe, profile_name, **kwargs):
        import_or_raise("pyarrow", PYARROW_IMPORT_ERROR_MESSAGE)
        if self.filename is not None and _is_dask_dataframe(dataframe):
            raise ValueError(
                "Writing a Dask dataframe to parquet with a filename specified is not supported"
            )
        if self.filename is not None and _is_spark_dataframe(dataframe):
            raise ValueError(
                "Writing a Spark dataframe to parquet with a filename specified is not supported"
            )
        self.kwargs["engine"] = "pyarrow"
        return super().serialize(dataframe, profile_name, **kwargs)

    def write_dataframe(self):
        file = self._get_filename()
        dataframe = clean_latlong(self.dataframe)
        dataframe.to_parquet(file, **self.kwargs)
