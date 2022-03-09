from woodwork.serializers import (
    ArrowSerializer,
    CSVSerializer,
    FeatherSerializer,
    OrcSerializer,
    ParquetSerializer,
    PickleSerializer,
)

# Dictionary used to get the corret serializer from a given format
FORMAT_TO_SERIALIZER = {
    CSVSerializer.format: CSVSerializer,
    PickleSerializer.format: PickleSerializer,
    ParquetSerializer.format: ParquetSerializer,
    ArrowSerializer.format: ArrowSerializer,
    FeatherSerializer.format: FeatherSerializer,
    OrcSerializer.format: OrcSerializer,
}


def get_serializer(format=None):
    """Get serializer class based on format"""
    serializer = FORMAT_TO_SERIALIZER.get(format)
    if serializer is None:
        error = "must be one of the following formats: {}"
        raise ValueError(error.format(", ".join(FORMAT_TO_SERIALIZER.keys())))

    return serializer
