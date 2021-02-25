=============
API Reference
=============

WoodworkTableAccessor
=====================

.. currentmodule:: woodwork.table_accessor
.. autosummary::
    :toctree: generated/

    WoodworkTableAccessor
    WoodworkTableAccessor.describe
    WoodworkTableAccessor.describe_dict
    WoodworkTableAccessor.iloc
    WoodworkTableAccessor.init
    WoodworkTableAccessor.loc
    WoodworkTableAccessor.mutual_information_dict
    WoodworkTableAccessor.mutual_information
    WoodworkTableAccessor.pop
    WoodworkTableAccessor.select
    WoodworkTableAccessor.set_index
    WoodworkTableAccessor.set_types
    WoodworkTableAccessor.to_csv
    WoodworkTableAccessor.to_dictionary
    WoodworkTableAccessor.to_parquet
    WoodworkTableAccessor.to_pickle
    WoodworkTableAccessor.value_counts

WoodworkColumnAccessor
======================

.. currentmodule:: woodwork.column_accessor
.. autosummary::
    :toctree: generated/

    WoodworkColumnAccessor
    WoodworkColumnAccessor.add_semantic_tags
    WoodworkColumnAccessor.iloc
    WoodworkColumnAccessor.init
    WoodworkColumnAccessor.loc
    WoodworkColumnAccessor.remove_semantic_tags
    WoodworkColumnAccessor.reset_semantic_tags
    WoodworkColumnAccessor.set_logical_type
    WoodworkColumnAccessor.set_semantic_tags

Schema
======

.. currentmodule:: woodwork.schema
.. autosummary::
    :toctree: generated/

    Schema
    Schema.add_semantic_tags
    Schema.remove_semantic_tags
    Schema.reset_semantic_tags
    Schema.set_index
    Schema.set_time_index
    Schema.set_types

Serialization
=============

.. currentmodule:: woodwork.serialize_accessor
.. autosummary::
    :toctree: generated/
    
    typing_info_to_dict
    write_dataframe
    write_typing_info
    write_woodwork_table

Deserialization
===============

.. currentmodule:: woodwork.deserialize_accessor
.. autosummary::
    :toctree: generated/

    read_table_typing_information
    read_woodwork_table

DataTable
=========

.. currentmodule:: woodwork.datatable
.. autosummary::
    :toctree: generated/

    DataTable
    DataTable.add_semantic_tags
    DataTable.describe
    DataTable.describe_dict
    DataTable.drop
    DataTable.head
    DataTable.iloc
    DataTable.mutual_information
    DataTable.mutual_information_dict
    DataTable.pop
    DataTable.remove_semantic_tags
    DataTable.rename
    DataTable.reset_semantic_tags
    DataTable.select
    DataTable.set_index
    DataTable.set_time_index
    DataTable.set_types
    DataTable.shape
    DataTable.to_csv
    DataTable.to_dataframe
    DataTable.to_parquet
    DataTable.to_pickle
    DataTable.update_dataframe
    DataTable.value_counts

DataColumn
==========

.. currentmodule:: woodwork.datacolumn
.. autosummary::
    :toctree: generated/

    DataColumn
    DataColumn.add_semantic_tags
    DataColumn.iloc
    DataColumn.remove_semantic_tags
    DataColumn.reset_semantic_tags
    DataColumn.set_logical_type
    DataColumn.set_semantic_tags
    DataColumn.shape
    DataColumn.to_series

Logical Types
=============

.. currentmodule:: woodwork.logical_types
.. autosummary::
    :toctree: generated/

    Boolean
    Categorical
    CountryCode
    Datetime
    Double
    EmailAddress
    Filepath
    FullName
    Integer
    IPAddress
    LatLong
    NaturalLanguage
    Ordinal
    PhoneNumber
    SubRegionCode
    Timedelta
    URL
    ZIPCode

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

    get_valid_mi_types
    read_csv

.. currentmodule:: woodwork.accessor_utils

.. autosummary::
    :toctree: generated
    :nosignatures:

    init_series

Demo Data
=========

.. currentmodule:: woodwork.demo
.. autosummary::
    :toctree: generated/

    load_retail
