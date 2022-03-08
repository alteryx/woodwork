from woodwork.serializers.serializer_base import (
    PYARROW_IMPORT_ERROR_MESSAGE,
    Serializer,
    clean_latlong,
)
from woodwork.utils import import_or_raise


class FeatherSerializer(Serializer):
    """Serialize a Woodwork table to disk as a feather file."""

    format = "feather"

    def serialize(self, dataframe, profile_name, **kwargs):
        import_or_raise("pyarrow", PYARROW_IMPORT_ERROR_MESSAGE)
        return super().serialize(dataframe, profile_name, **kwargs)

    def write_dataframe(self):
        file = self._get_filename()
        dataframe = clean_latlong(self.dataframe)
        dataframe.to_feather(file, **self.kwargs)
