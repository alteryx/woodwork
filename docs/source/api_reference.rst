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
    WoodworkTableAccessor.describe
    WoodworkTableAccessor.describe_dict
    WoodworkTableAccessor.drop
    WoodworkTableAccessor.iloc
    WoodworkTableAccessor.index
    WoodworkTableAccessor.init
    WoodworkTableAccessor.init_with_full_schema
    WoodworkTableAccessor.init_with_partial_schema
    WoodworkTableAccessor.loc
    WoodworkTableAccessor.logical_types
    WoodworkTableAccessor.metadata
    WoodworkTableAccessor.mutual_information
    WoodworkTableAccessor.mutual_information_dict
    WoodworkTableAccessor.name
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
    WoodworkTableAccessor.time_index
    WoodworkTableAccessor.to_disk
    WoodworkTableAccessor.to_dictionary
    WoodworkTableAccessor.types
    WoodworkTableAccessor.use_standard_tags
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
    WoodworkColumnAccessor.remove_semantic_tags
    WoodworkColumnAccessor.reset_semantic_tags
    WoodworkColumnAccessor.semantic_tags
    WoodworkColumnAccessor.set_logical_type
    WoodworkColumnAccessor.set_semantic_tags
    WoodworkColumnAccessor.use_standard_tags

TableSchema
===========

.. currentmodule:: woodwork.table_schema
.. autosummary::
    :toctree: generated/

    TableSchema
    TableSchema.add_semantic_tags
    TableSchema.index
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
    ColumnSchema.description
    ColumnSchema.origin
    ColumnSchema.is_boolean
    ColumnSchema.is_categorical
    ColumnSchema.is_datetime
    ColumnSchema.is_numeric
    ColumnSchema.metadata
    
Serialization
=============

.. currentmodule:: woodwork.serialize
.. autosummary::
    :toctree: generated/
    
    typing_info_to_dict
    write_dataframe
    write_typing_info
    write_woodwork_table

Deserialization
===============

.. currentmodule:: woodwork.deserialize
.. autosummary::
    :toctree: generated/

    read_table_typing_information
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
    read_file

.. currentmodule:: woodwork.accessor_utils

.. autosummary::
    :toctree: generated
    :nosignatures:

    get_invalid_schema_message
    init_series
    is_schema_valid

Demo Data
=========

.. currentmodule:: woodwork.demo
.. autosummary::
    :toctree: generated/

    load_retail
