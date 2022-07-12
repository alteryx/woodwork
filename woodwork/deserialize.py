from woodwork.deserializers.utils import _get_deserializer


def from_disk(
    path,
    filename=None,
    data_subdirectory="data",
    typing_info_filename="woodwork_typing_info.json",
    profile_name=None,
    validate=False,
    **kwargs,
):
    """Convenience function to call `read_woodwork_table`."""
    deserialized_df = read_woodwork_table(
        path,
        filename=filename,
        data_subdirectory=data_subdirectory,
        typing_info_filename=typing_info_filename,
        profile_name=profile_name,
        validate=validate,
        **kwargs,
    )

    return deserialized_df


def read_woodwork_table(
    path,
    filename=None,
    data_subdirectory="data",
    typing_info_filename="woodwork_typing_info.json",
    profile_name=None,
    validate=False,
    format=None,
    **kwargs,
):
    """Read Woodwork table from disk, S3 path, or URL.

    Args:
        path (str): Directory on disk, S3 path, or URL to read data and typing information.
        filename (str, optional): The name of the file used to store the data during serialization. If not specified, will be
            determined from the typing info file. Must be specified when deserializing from a single parquet file.
        data_subdirectory (str, optional): The subdirectory in which the data was stored during serialization. Defaults to "data".
        typing_info_filename (str, optional): The name of the JSON file used to store the Woodwork typing information during
            serialization. Defaults to "woodwork_typing_info.json".
        format (str, optional): The format used to serialize the data. Required if the serialized filename suffix does not
            match the format or when deserializing from parquet files into Dask or Spark dataframes.
        profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
            Set to False to use an anonymous profile.
        validate (bool, optional): Whether parameter and data validation should occur when initializing Woodwork dataframe
            during deserialization. Defaults to False. Note: If serialized data was modified outside of Woodwork and you
            are unsure of the validity of the data or typing information, `validate` should be set to True.
        kwargs (keywords): Additional keyword arguments to pass as keyword arguments to the underlying deserialization method.

    Returns:
        DataFrame: DataFrame with Woodwork typing information initialized.
    """
    deserializer = _get_deserializer(
        path,
        filename,
        data_subdirectory,
        typing_info_filename,
        profile_name,
        format,
    )
    return deserializer.deserialize(profile_name, validate, **kwargs)
