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

    import woodwork as ww

    data = ww.demo.load_retail(nrows=100)

    data.head(5)

As we can see, this is a DataFrame containing different data types. Let's use Woodwork to create a DataTable.

.. ipython:: python

    dt = ww.DataTable(data, name="retail")

    dt.types

Woodwork was able to infer the Logical Types present in our data on the DataFrame dtypes. In addition, it also added semantic tags to some of the columns.

Let's change some of the columns to a different Logical Type


.. ipython:: python

    dt.set_logical_types({
        'quantity': 'Categorical',
        'customer_name': 'Categorical',
        'country': 'Categorical'
    })

    dt.types

We can also select some of the columns based on the Logical Type

.. ipython:: python

    numeric_dt = dt.select_ltypes(['WholeNumber', 'Double'])

    numeric_dt.types

Let's add some Semantic Tags to some of the columns

.. ipython:: python

    dt.set_semantic_tags({'order_id':'order_index', 'order_date': 'order_time_index'})

    dt.types


We can also select some of the columns based on a Semantic Tag

.. ipython:: python

    category_dt = dt.select_semantic_tags('category')

    category_dt.types

We can also select with mutiple Semantic Tags

.. ipython:: python

    category_numeric_dt = dt.select_semantic_tags(['numeric', 'category'])

    category_numeric_dt.types


If we wanted to select individual columns, we just need to specify the column name. We can then get access to the data in the Data Column

.. ipython:: python

    dc = dt['total']

    dc

    dc.series


We can also remove some of the Semnatic tags on the DataColumn

.. ipython:: python

    dt.remove_semantic_tags({'order_id':'order_index'})

    dt.types


Notice how the index column has now been removed. If we wanted to remove all user-added semantic tags, we can also do that

.. ipython:: python

    dt.reset_semantic_tags()

    dt.types
