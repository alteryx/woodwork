from woodwork.deserializers import get_deserializer


def read_woodwork_table(
    path,
    filename=None,
    data_subdirectory="data",
    typing_info_file="woodwork_typing_info.json",
    profile_name=None,
    validate=False,
    **kwargs,
):
    """Read Woodwork table from disk, S3 path, or URL.

    Args:
        path (str): Directory on disk, S3 path, or URL to read `woodwork_typing_info.json`.
        filename (str, optional): The name of the file used to store the data during serialization. If not specified, will be
            determined from the typing info file.
        data_subdirectory (str, optional): The subdirectory in which the data was stored during serialization. Defaults to "data".
        typing_info_file (str, optional): The name of the JSON file used to store the Woodwork typing information during
            serialization. Defaults to "woodwork_typing_info.json".
        profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
            Set to False to use an anonymous profile.
        validate (bool, optional): Whether parameter and data validation should occur when initializing Woodwork dataframe
            during deserialization. Defaults to False. Note: If serialized data was modified outside of Woodwork and you
            are unsure of the validity of the data or typing information, `validate` should be set to True.
        kwargs (keywords): Additional keyword arguments to pass as keyword arguments to the underlying deserialization method.

    Returns:
        DataFrame: DataFrame with Woodwork typing information initialized.
    """
    deserializer = get_deserializer(
        path, filename, data_subdirectory, typing_info_file, profile_name
    )
    return deserializer.deserialize(profile_name, validate, **kwargs)
