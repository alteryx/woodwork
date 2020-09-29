.. _changelog:

Changelog
---------
**Future Release**
    * Enhancements
        * Implement setitem on DataTable to create/overwrite an existing DataColumn (:pr:`165`)
    * Fixes
    * Changes
    * Documentation Changes
        * Remove unused code from sphinx conf.py (:pr:`160`)
        * Update README and docs with new Woodwork logo (:pr:`161`)
        * Add Github URL to theme conf (:pr:`163`)
        * Add DataTable and DataColumn to API Reference (:pr:`162`)
    * Testing Changes
        * Add tests that confirm dtypes are as expected after DataTable init (:pr:`152`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`thehomebrewnerd`, :user:`tamargrey`

**v0.0.2** September 28, 2020
    * Fixes
        * Fix formatting issue when printing global config variables (:pr:`138`)
    * Changes
        * Change add_standard_tags to use_standard_Tags to better describe behavior (:pr:`149`)
        * Change access of underlying dataframe to be through ``to_pandas`` with ._dataframe field on class (:pr:`146`)
        * Remove ``replace_none`` parameter to DataTables (:pr:`146`)
    * Documentation Changes
        * Add working code example to README and create Using Woodwork page (:pr:`103`)
    * Testing Changes

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

**v0.1.0** September 24, 2020
    * Add ``natural_language_threshold`` global config option used for Categorical/NaturalLanguage type inference (:pr:`135`)
    * Add global config options and add ``datetime_format`` option for type inference (:pr:`134`)
    * Fix bug with Integer and WholeNumber inference in column with ``pd.NA`` values (:pr:`133`)
    * Add DataTable.ltypes property to return series of logical types (:pr:`131`)
    * Add ability to create new datatable from specified columns with ``dt[[columns]]`` (:pr:`127`)
    * Handle setting and tagging of index and time index columns (:pr:`125`)
    * Add combined tag and ltype selection (:pr:`124`)
    * Add changelog, and update changelog check to CI (:pr:`123`)
    * Implement reset_semantic_tags (:pr:`118`)
    * Implement DataTable getitem (:pr:`119`)
    * Add remove_semantic_tags method (:pr:`117`)
    * Add semantic tag selection (:pr:`106`)
    * Add github action, rename to woodwork (:pr:`113`)
    * Add license to setup.py (:pr:`112`)
    * Reset semantic tags on logical type change (:pr:`107`)
    * Add standard numeric and category tags (:pr:`100`)
    * Change semantic_types to semantic_tags, a set of strings (:pr:`100`)
    * Update dataframe dtypes based on logical types (:pr:`94`)
    * Add select_logical_types to DataTable (:pr:`96`)
    * Add pygments to dev-requirements.txt (:pr:`97`)
    * Add replacing None with np.nan in DataTable init (:pr:`87`)
    * Refactor DataColumn to make semantic_types and logical_type private (:pr:`86`)
    * Add pandas_dtype to each Logical Type, and remove dtype attribute on DataColumn (:pr:`85`)
    * Add set_semantic_types methods on both DataTable and DataColumn (:pr:`75`)
    * Support passing camel case or snake case strings for setting logical types (:pr:`74`)
    * Improve flexibility when setting semantic types (:pr:`72`)
    * Add Whole Number Inference of Logical Types (:pr:`66`)
    * Add dtypes property to DataTables and repr for DataColumn (:pr:`61`)
    * Allow specification of semantic types during DataTable creation (:pr:`69`)
    * Implements set_logical_types on DataTable (:pr:`65`)
    * Add init files to tests to fix code coverage (:pr:`60`)
    * Add AutoAssign bot (:pr:`59`)
    * Add logical types validation in DataTables (:pr:`49`)
    * Fix working_directory in CI (:pr:`57`)
    * Add infer_logical_types for DataColumn (:pr:`45`)
    * Fix ReadME library name, and code coverage badge (:pr:`56`, :pr:`56`)
    * Add code coverage (:pr:`51`)
    * Improve and refactor the validation checks during initialization of a DataTable (:pr:`40`)
    * Add dataframe attribute to DataTable (:pr:`39`)
    * Update ReadME with minor usage details (:pr:`37`)
    * Add License (:pr:`34`)
    * Rename from datatables to data_tables (:pr:`4`)
    * Add Logical Types, DataTable, DataColumn (:pr:`3`)
    * Add Makefile, setup.py, requirements.txt (:pr:`2`)
    * Initial Release (:pr:`1`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

.. command
.. git log --pretty=oneline --abbrev-commit
