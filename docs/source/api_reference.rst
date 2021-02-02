=============
API Reference
=============

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

Demo Data
=========

.. currentmodule:: woodwork.demo
.. autosummary::
    :toctree: generated/

    load_retail
