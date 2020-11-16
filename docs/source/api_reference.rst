=============
API Reference
=============

DataTable
=========

.. currentmodule:: woodwork.datatable
.. autosummary::
    :toctree: generated/

    DataTable
    DataTable.shape
    DataTable.add_semantic_tags
    DataTable.remove_semantic_tags
    DataTable.reset_semantic_tags
    DataTable.select
    DataTable.iloc
    DataTable.set_index
    DataTable.set_logical_types
    DataTable.set_semantic_tags
    DataTable.set_time_index
    DataTable.to_dataframe
    DataTable.describe
    DataTable.mutual_information
    DataTable.value_counts
    DataTable.to_csv
    DataTable.to_pickle
    DataTable.to_parquet
    DataTable.rename


DataColumn
==========

.. currentmodule:: woodwork.datacolumn
.. autosummary::
    :toctree: generated/

    DataColumn
    DataColumn.shape
    DataColumn.iloc
    DataColumn.add_semantic_tags
    DataColumn.remove_semantic_tags
    DataColumn.reset_semantic_tags
    DataColumn.set_logical_type
    DataColumn.set_semantic_tags
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
    list_semantic_tags
    read_csv

Demo Data
=========

.. currentmodule:: woodwork.demo
.. autosummary::
    :toctree: generated/

    load_retail
