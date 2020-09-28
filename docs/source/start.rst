Start
*****

.. ipython:: python
    :suppress:

    import urllib.request

    opener = urllib.request.build_opener()
    opener.addheaders = [("Testing", "True")]
    urllib.request.install_opener(opener)



In this guide, we will walk through an example of creating a Woodwork DataTable, and will show how to update and remove logical types and semantic tags. We will also demonstrate how to use the typing information to select subsets of data.

First, let's load in some demo retail data.

.. ipython:: python

    import woodwork as ww

    data = ww.demo.load_retail(nrows=100)

    data.head(5)

As we can see, this is a dataframe containing several different data types, including dates, categorical values, numeric values and natural language descriptions. Let's use Woodwork to create a DataTable from this data.

Creating a DataTable
====================
Creating a Woodwork DataTable is as simple as passing in a dataframe with the data of interest during initialization. An optional name parameter can be specified to label the DataTable.

.. ipython:: python

    dt = ww.DataTable(data, name="retail")

    dt.types

Using just this simple call, Woodwork was able to infer the logical types present in our data by analyzing the dataframe dtypes as well as the information contained in the columns. In addition, Woodwork also added semantic tags to some of the columns based on the logical types that were inferred.


Updating Logical Types
======================
If the initial inference was not to our liking, the logical type can be changed to a more appropriate value. Let's change some of the columns to a different logical type to illustrate this process. Below we will set the logical type for the ``quantity``, ``customer_name`` and ``country`` columns to be ``Categorical``.


.. ipython:: python

    dt.set_logical_types({
        'quantity': 'Categorical',
        'customer_name': 'Categorical',
        'country': 'Categorical'
    })

    dt.types

If we now inspect the information in the `types` output, we can see that the Logical type for the three columns has been updated with the `Categorical` logical type we specified.

Selecting Columns
=================
Now that we have logical types we are happy with, we can select a subset of the columns based on their logical types. Let's select only the columns that have a logical type of ``WholeNumber`` or ``Double``:

.. ipython:: python

    numeric_dt = dt.select_ltypes(['WholeNumber', 'Double'])

    numeric_dt.types

This selection process has returned a new ``DataTable`` containing only the columns that match the logical types we specified. After we have selected the columns we want, we can also access a dataframe containing just those columns if we need it for additional analysis.

.. ipython:: python

    numeric_dt.to_pandas()

.. note::
    Accessing the dataframe associated with a ``DataTable`` by using ``dt.to_pandas()`` will return a reference to the dataframe. Modifications to the returned dataframe can cause unexpected results. If you need to modify the dataframe, you should use ``dt.to_pandas(copy=True)`` to return a copy of the stored dataframe that can be safely modified without impacting the ``DataTable`` behavior.

Adding Semantic Tags
====================
Next, let's add semantic tags to some of the columns. We will add the tag of ``product_details`` to the ``description`` column and tag the ``total`` column with ``currency``.

.. ipython:: python

    dt.set_semantic_tags({'description':'product_details', 'total': 'currency'})

    dt.types


We can also select columns based on a semantic tag. Perhaps we want to only select the columns tagged with ``category``:

.. ipython:: python

    category_dt = dt.select_semantic_tags('category')

    category_dt.types

We can also select columns using mutiple semantic tags, or even a mixture of semantic tags and logical types:

.. ipython:: python

    category_numeric_dt = dt.select_semantic_tags(['numeric', 'category'])

    category_numeric_dt.types

    mixed_dt = dt.select(include=['Boolean', 'product_details'])

    mixed_dt.types


If we wanted to select an individual column, we just need to specify the column name. We can then get access to the data in the DataColumn using the ``series`` attribute:

.. ipython:: python

    dc = dt['total']

    dc

    dc.series


You can also access multiple columns by supplying a list of column names:

.. ipython:: python

   multiple_cols_dt = dt[['product_id', 'total', 'unit_price']]

   multiple_cols_dt.types


Removing Semantic Tags
======================
We can also remove specific semantic tags from a column if they are no longer needed. Let's remove the ``product_details`` tag from the ``description`` column:

.. ipython:: python

    dt.remove_semantic_tags({'description':'product_details'})

    dt.types

Notice how the ``product_details`` tag has now been removed from the ``description`` column. If we wanted to remove all user-added semantic tags from all columns, we can also do that:

.. ipython:: python

    dt.reset_semantic_tags()

    dt.types
