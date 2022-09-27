=============
API Reference
=============

WoodworkTableAccessor
=====================

.. currentmodule:: woodwork.table_accessor
.. autosummary::
    :toctree: generated/

    WoodworkTableAccessor
    WoodworkTableAccessor.add_semantic_tags
    WoodworkTableAccessor.dependence
    WoodworkTableAccessor.dependence_dict
    WoodworkTableAccessor.describe
    WoodworkTableAccessor.describe_dict
    WoodworkTableAccessor.drop
    WoodworkTableAccessor.iloc
    WoodworkTableAccessor.index
    WoodworkTableAccessor.infer_temporal_frequencies
    WoodworkTableAccessor.init
    WoodworkTableAccessor.init_with_full_schema
    WoodworkTableAccessor.init_with_partial_schema
    WoodworkTableAccessor.loc
    WoodworkTableAccessor.logical_types
    WoodworkTableAccessor.metadata
    WoodworkTableAccessor.mutual_information
    WoodworkTableAccessor.mutual_information_dict
    WoodworkTableAccessor.name
    WoodworkTableAccessor.pearson_correlation
    WoodworkTableAccessor.pearson_correlation_dict
    WoodworkTableAccessor.physical_types
    WoodworkTableAccessor.pop
    WoodworkTableAccessor.remove_semantic_tags
    WoodworkTableAccessor.rename
    WoodworkTableAccessor.reset_semantic_tags
    WoodworkTableAccessor.schema
    WoodworkTableAccessor.select
    WoodworkTableAccessor.semantic_tags
    WoodworkTableAccessor.set_index
    WoodworkTableAccessor.set_time_index
    WoodworkTableAccessor.set_types
    WoodworkTableAccessor.spearman_correlation
    WoodworkTableAccessor.spearman_correlation_dict
    WoodworkTableAccessor.time_index
    WoodworkTableAccessor.to_disk
    WoodworkTableAccessor.to_dictionary
    WoodworkTableAccessor.types
    WoodworkTableAccessor.use_standard_tags
    WoodworkTableAccessor.validate_logical_types
    WoodworkTableAccessor.value_counts

WoodworkColumnAccessor
======================

.. currentmodule:: woodwork.column_accessor
.. autosummary::
    :toctree: generated/

    WoodworkColumnAccessor
    WoodworkColumnAccessor.add_semantic_tags
    WoodworkColumnAccessor.box_plot_dict
    WoodworkColumnAccessor.description
    WoodworkColumnAccessor.origin
    WoodworkColumnAccessor.iloc
    WoodworkColumnAccessor.init
    WoodworkColumnAccessor.loc
    WoodworkColumnAccessor.logical_type
    WoodworkColumnAccessor.metadata
    WoodworkColumnAccessor.nullable
    WoodworkColumnAccessor.remove_semantic_tags
    WoodworkColumnAccessor.reset_semantic_tags
    WoodworkColumnAccessor.semantic_tags
    WoodworkColumnAccessor.set_logical_type
    WoodworkColumnAccessor.set_semantic_tags
    WoodworkColumnAccessor.use_standard_tags
    WoodworkColumnAccessor.validate_logical_type

TableSchema
===========

.. currentmodule:: woodwork.table_schema
.. autosummary::
    :toctree: generated/

    TableSchema
    TableSchema.add_semantic_tags
    TableSchema.index
    TableSchema.get_subset_schema
    TableSchema.logical_types
    TableSchema.metadata
    TableSchema.rename
    TableSchema.remove_semantic_tags
    TableSchema.reset_semantic_tags
    TableSchema.name
    TableSchema.semantic_tags
    TableSchema.set_index
    TableSchema.set_time_index
    TableSchema.set_types
    TableSchema.time_index
    TableSchema.types
    TableSchema.use_standard_tags

ColumnSchema
============

.. currentmodule:: woodwork.table_schema
.. autosummary::
    :toctree: generated/

    ColumnSchema
    ColumnSchema.custom_tags
    ColumnSchema.description
    ColumnSchema.origin
    ColumnSchema.is_boolean
    ColumnSchema.is_categorical
    ColumnSchema.is_datetime
    ColumnSchema.is_numeric
    ColumnSchema.metadata

Serialization
=============

.. currentmodule:: woodwork.serializers.serializer_base
.. autosummary::
    :toctree: generated/

    typing_info_to_dict

Deserialization
===============

.. currentmodule:: woodwork.deserialize
.. autosummary::
    :toctree: generated/

    from_disk
    read_woodwork_table

Logical Types
=============

.. currentmodule:: woodwork.logical_types
.. autosummary::
    :toctree: generated/

    Address
    Age
    AgeFractional
    AgeNullable
    Boolean
    BooleanNullable
    Categorical
    CountryCode
    CurrencyCode
    Datetime
    Double
    EmailAddress
    Filepath
    Integer
    IntegerNullable
    IPAddress
    LatLong
    NaturalLanguage
    Ordinal
    PersonFullName
    PhoneNumber
    PostalCode
    SubRegionCode
    Timedelta
    Unknown
    URL

TypeSystem
==========

.. currentmodule:: woodwork.type_sys.type_system
.. autosummary::
    :toctree: generated/

    TypeSystem
    TypeSystem.add_type
    TypeSystem.infer_logical_type
    TypeSystem.remove_type
    TypeSystem.reset_defaults
    TypeSystem.update_inference_function
    TypeSystem.update_relationship

Utils
=====

.. currentmodule:: woodwork.type_sys.utils

Type Utils
~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    list_logical_types
    list_semantic_tags

.. currentmodule:: woodwork.utils

General Utils
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    concat_columns
    get_valid_mi_types
    get_valid_pearson_types
    get_valid_spearman_types
    read_file

.. currentmodule:: woodwork.accessor_utils

.. autosummary::
    :toctree: generated
    :nosignatures:

    get_invalid_schema_message
    init_series
    is_schema_valid

Statistics Utils
~~~~~~~~~~~~~~~~

.. currentmodule:: woodwork.statistics_utils

.. autosummary::
    :toctree: generated
    :nosignatures:

    infer_frequency

Demo Data
=========

.. currentmodule:: woodwork.demo
.. autosummary::
    :toctree: generated/

    load_retail
