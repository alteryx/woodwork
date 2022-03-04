from mimetypes import add_type, guess_type

from woodwork.serializers import (
    ArrowSerializer,
    CSVSerializer,
    FeatherSerializer,
    OrcSerializer,
    ParquetSerializer,
    PickleSerializer,
)

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
