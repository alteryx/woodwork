.. featuretools documentation main file, created by
   sphinx-quickstart on Thu May 19 20:40:30 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. currentmodule:: featuretools


What is Woodwork?
-----------------

Woodwork is a library that helps with data typing of 2-dimensional tabular data structures.
It provides a DataTable object, which contains the physical, logical, and semantic data types.
It can be used with `Featuretools <https://www.featuretools.com>`__, `EvalML <https://evalml.featurelabs.com/en/latest/>`__, and general machine learning applications where logical and semantic typing information is important.


Quick Start
-----------

Below is an example of using a Woodwork DataTable to automatically infer the Logical Types for a data structure.


.. ipython:: python
    :suppress:

    import urllib.request

    opener = urllib.request.build_opener()
    opener.addheaders = [("Testing", "True")]
    urllib.request.install_opener(opener)


.. ipython:: python

    import woodwork as ww

    data = ww.demo.load_retail(nrows=100)

    dt = ww.DataTable(data, name="retail")
    dt.types

Table of contents
-----------------

.. toctree::
   :maxdepth: 1

   install
   start
   api_reference
   changelog
