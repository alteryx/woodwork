Start
*****

.. ipython:: python
    :suppress:

    import urllib.request

    opener = urllib.request.build_opener()
    opener.addheaders = [("Testing", "True")]
    urllib.request.install_opener(opener)



First, let's load in some demo retail data.

.. ipython:: python

    import data_tables as dt

    data = dt.demo.load_retail(nrows=100)

    data.head(5)

    data.dtypes


As we can see, this is a DataFrame containing different data types. Let's use <TBD> to create a DataTable.

.. ipython:: python

    dt = dt.DataTable(data, name="retail")

    dt.types

<TBD> was able to infer the Logical Types present in our data on the DataFrame dtypes. In addition, it also added semantic tags to some of the columns.

Let's change some of the columns to a different Logical Type.


.. ipython:: python

    dt.set_logical_types({
        'quantity': 'Categorical',
        'customer_name': 'Categorical',
        'country': 'Categorical'
    })

    dt.types

We can also select some of the columns based on the Logical Type.

.. ipython:: python

    numeric_dt = dt.select_ltypes(['WholeNumber', 'Integer', 'Double'])

    numeric_dt.types


We can also select some of the columns based on the Semantic Tag. TODO
