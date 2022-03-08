from woodwork.serializers import FeatherSerializer


class ArrowSerializer(FeatherSerializer):
    """Serialize a Woodwork table to disk as an arrow file."""

    format = "arrow"
