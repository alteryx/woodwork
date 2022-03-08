from woodwork.accessor_utils import _is_dask_dataframe
from woodwork.serializer_utils import clean_latlong, save_orc_file
from woodwork.serializers.serializer_base import (
    PYARROW_IMPORT_ERROR_MESSAGE,
    Serializer,
)
from woodwork.utils import import_or_raise


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
        file = self._get_filename()
        dataframe = clean_latlong(self.dataframe)
        save_orc_file(dataframe, file)
