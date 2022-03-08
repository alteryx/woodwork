from woodwork.deserializers import FeatherDeserializer


class ArrowDeserializer(FeatherDeserializer):
    """Deserialize Woodwork table from serialized data in arrow format."""

    format = "arrow"
