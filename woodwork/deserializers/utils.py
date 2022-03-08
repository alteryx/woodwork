from woodwork.serializer_utils import read_table_typing_information

from woodwork.deserializers import (
    ArrowDeserializer,
    CSVDeserializer,
    FeatherDeserializer,
    OrcDeserializer,
    ParquetDeserializer,
    PickleDeserializer,
)

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
    typing_info = read_table_typing_information(
        path, typing_info_filename, profile_name
    )
    format = typing_info["loading_info"]["type"]

    deserializer_cls = FORMAT_TO_DESERIALIZER.get(format)

    return deserializer_cls(path, filename, data_subdirectory, typing_info)


