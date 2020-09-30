=============
API Reference
=============

DataTable
=========

.. currentmodule:: woodwork.data_table
.. autosummary::
    :toctree: generated/

    DataTable
    DataTable.add_semantic_tags
    DataTable.remove_semantic_tags
    DataTable.reset_semantic_tags
    DataTable.select
    DataTable.select_ltypes
    DataTable.select_semantic_tags
    DataTable.set_index
    DataTable.set_logical_types
    DataTable.set_semantic_tags
    DataTable.set_time_index
    DataTable.to_pandas
    DataTable.describe


DataColumn
==========

.. currentmodule:: woodwork.data_column
.. autosummary::
    :toctree: generated/

    DataColumn
    DataColumn.add_semantic_tags
    DataColumn.remove_semantic_tags
    DataColumn.reset_semantic_tags
    DataColumn.set_logical_type
    DataColumn.set_semantic_tags
    DataColumn.to_pandas


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
    Integer
    EmailAddress
    Filepath
    FullName
    IPAddress
    LatLong
    NaturalLanguage
    Ordinal
    PhoneNumber
    SubRegionCode
    Timedelta
    URL
    WholeNumber
    ZIPCode

.. currentmodule:: woodwork.utils

Utils
=====

General Utils
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    list_logical_types