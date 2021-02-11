.. _release_notes:

Release Notes
-------------
**Future Release**
    * Enhancements
    * Fixes
    * Changes
        * Avoid calculating mutualinfo for non-unique columns (:pr:`563`)
    * Documentation Changes
    * Testing Changes
        * Update branch reference in tests workflow (:pr:`552`)
        * Fixed text on back arrow on install page (:pr:`564`)
        * Refactor test_datatable.py (:pr:`574`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`, :user:`johnbridstrup`

**v0.0.9 February 5, 2021**
    * Enhancements
        * Add Python 3.9 support without Koalas testing (:pr:`511`)
        * Add ``get_valid_mi_types`` function to list LogicalTypes valid for mutual information calculation (:pr:`517`)
    * Fixes
        * Handle missing values in Datetime columns when calculating mutual information (:pr:`516`)
        * Support numpy 1.20.0 by restricting version for koalas and changing serialization error message (:pr:`532`)
        * Move Koalas option setting to DataTable init instead of import (:pr:`543`)
    * Documentation Changes
        * Add Alteryx OSS Twitter link (:pr:`519`)
        * Update logo and add new favicon (:pr:`521`)
        * Multiple improvements to Getting Started page and guides (:pr:`527`)
        * Clean up API Reference and docstrings (:pr:`536`)
        * Added Open Graph for Twitter and Facebook (:pr:`544`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

**v0.0.8 January 25, 2021**
    * Enhancements
        * Add ``DataTable.df`` property for accessing the underling DataFrame (:pr:`470`)
        * Set index of underlying DataFrame to match DataTable index (:pr:`464`)
    * Fixes
        * Sort underlying series when sorting dataframe (:pr:`468`)
        * Allow setting indices to current index without side effects (:pr:`474`)
    * Changes
       * Fix release document with Github Actions link for CI (:pr:`462`)
       * Don't allow registered LogicalTypes with the same name (:pr:`477`)
       * Move ``str_to_logical_type`` to TypeSystem class (:pr:`482`)
       * Remove ``pyarrow`` from core dependencies (:pr:`508`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

**v0.0.7 December 14, 2020**
    * Enhancements
        * Allow for user-defined logical types and inference functions in TypeSystem object (:pr:`424`)
        * Add ``__repr__`` to DataTable (:pr:`425`)
        * Allow initializing DataColumn with numpy array (:pr:`430`)
        * Add ``drop`` to DataTable (:pr:`434`)
        * Migrate CI tests to Github Actions (:pr:`417`, :pr:`441`, :pr:`451`)
        * Add ``metadata`` to DataColumn for user-defined metadata (:pr:`447`)
    * Fixes
        * Update DataColumn name when using setitem on column with no name (:pr:`426`)
        * Don't allow pickle serialization for Koalas DataFrames (:pr:`432`)
        * Check DataTable metadata in equality check (:pr:`449`)
        * Propagate all attributes of DataTable in ``_new_dt_including`` (:pr:`454`)
    * Changes
        * Update links to use alteryx org Github URL (:pr:`423`)
        * Support column names of any type allowed by the underlying DataFrame (:pr:`442`)
        * Use ``object`` dtype for LatLong columns for easy access to latitude and longitude values (:pr:`414`)
        * Restrict dask version to prevent 2020.12.0 release from being installed (:pr:`453`)
        * Lower minimum requirement for numpy to 1.15.4, and set pandas minimum requirement 1.1.1 (:pr:`459`)
    * Testing Changes
        * Fix missing test coverage (:pr:`436`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`tamargrey`, :user:`thehomebrewnerd`

**v0.0.6 November 30, 2020**
    * Enhancements
        * Add support for creating DataTable from Koalas DataFrame (:pr:`327`)
        * Add ability to initialize DataTable with numpy array (:pr:`367`)
        * Add ``describe_dict`` method to DataTable (:pr:`405`)
        * Add ``mutual_information_dict`` method to DataTable (:pr:`404`)
        * Add ``metadata`` to DataTable for user-defined metadata (:pr:`392`)
        * Add ``update_dataframe`` method to DataTable to update underlying DataFrame (:pr:`407`)
        * Sort dataframe if ``time_index`` is specified, bypass sorting with ``already_sorted`` parameter. (:pr:`410`)
        * Add ``description`` attribute to DataColumn (:pr:`416`)
        * Implement ``DataColumn.__len__`` and ``DataTable.__len__`` (:pr:`415`)
    * Fixes
        * Rename ``data_column.py`` ``datacolumn.py`` (:pr:`386`)
        * Rename ``data_table.py`` ``datatable.py`` (:pr:`387`)
        * Rename ``get_mutual_information`` ``mutual_information`` (:pr:`390`)
    * Changes
        * Lower moto test requirement for serialization/deserialization (:pr:`376`)
        * Make Koalas an optional dependency installable with woodwork[koalas] (:pr:`378`)
        * Remove WholeNumber LogicalType from Woodwork (:pr:`380`)
        * Updates to LogicalTypes to support Koalas 1.4.0 (:pr:`393`)
        * Replace ``set_logical_types`` and ``set_semantic_tags`` with just ``set_types`` (:pr:`379`)
        * Remove ``copy_dataframe`` parameter from DataTable initialization (:pr:`398`)
        * Implement ``DataTable.__sizeof__`` to return size of the underlying dataframe (:pr:`401`)
        * Include Datetime columns in mutual info calculation (:pr:`399`)
        * Maintain column order on DataTable operations (:pr:`406`)
    * Testing Changes
        * Add pyarrow, dask, and koalas to automated dependency checks (:pr:`388`)
        * Use new version of pull request Github Action (:pr:`394`)
        * Improve parameterization for ``test_datatable_equality`` (:pr:`409`)

    Thanks to the following people for contributing to this release:
    :user:`ctduffy`, :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

**Breaking Changes**
    * The ``DataTable.set_semantic_tags`` method was removed. ``DataTable.set_types`` can be used instead.
    * The ``DataTable.set_logical_types`` method was removed. ``DataTable.set_types`` can be used instead.
    * ``WholeNumber`` was removed from LogicalTypes. Columns that were previously inferred as WholeNumber will now be inferred as Integer.
    * The ``DataTable.get_mutual_information`` was renamed to ``DataTable.mutual_information``.
    * The ``copy_dataframe`` parameter was removed from DataTable initialization.

**v0.0.5 November 11, 2020**
    * Enhancements
        * Add ``__eq__`` to DataTable and DataColumn and update LogicalType equality (:pr:`318`)
        * Add ``value_counts()`` method to DataTable (:pr:`342`)
        * Support serialization and deserialization of DataTables via csv, pickle, or parquet (:pr:`293`)
        * Add ``shape`` property to DataTable and DataColumn (:pr:`358`)
        * Add ``iloc`` method to DataTable and DataColumn (:pr:`365`)
        * Add ``numeric_categorical_threshold`` config value to allow inferring numeric columns as Categorical (:pr:`363`)
        * Add ``rename`` method to DataTable (:pr:`367`)
    * Fixes
        * Catch non numeric time index at validation (:pr:`332`)
    * Changes
        * Support logical type inference from a Dask DataFrame (:pr:`248`)
        * Fix validation checks and ``make_index`` to work with Dask DataFrames (:pr:`260`)
        * Skip validation of Ordinal order values for Dask DataFrames (:pr:`270`)
        * Improve support for datetimes with Dask input (:pr:`286`)
        * Update ``DataTable.describe`` to work with Dask input (:pr:`296`)
        * Update ``DataTable.get_mutual_information`` to work with Dask input (:pr:`300`)
        * Modify ``to_pandas`` function to return DataFrame with correct index (:pr:`281`)
        * Rename ``DataColumn.to_pandas`` method to ``DataColumn.to_series`` (:pr:`311`)
        * Rename ``DataTable.to_pandas`` method to ``DataTable.to_dataframe`` (:pr:`319`)
        * Remove UserWarning when no matching columns found (:pr:`325`)
        * Remove ``copy`` parameter from ``DataTable.to_dataframe`` and ``DataColumn.to_series`` (:pr:`338`)
        * Allow pandas ExtensionArrays as inputs to DataColumn (:pr:`343`)
        * Move warnings to a separate exceptions file and call via UserWarning subclasses (:pr:`348`)
        * Make Dask an optional dependency installable with woodwork[dask] (:pr:`357`)
    * Documentation Changes
        * Create a guide for using Woodwork with Dask (:pr:`304`)
        * Add conda install instructions (:pr:`305`, :pr:`309`)
        * Fix README.md badge with correct link (:pr:`314`)
        * Simplify issue templates to make them easier to use (:pr:`339`)
        * Remove extra output cell in Start notebook (:pr:`341`)
    * Testing Changes
        * Parameterize numeric time index tests (:pr:`288`)
        * Add DockerHub credentials to CI testing environment (:pr:`326`)
        * Fix removing files for serialization test (:pr:`350`)

    Thanks to the following people for contributing to this release:
    :user:`ctduffy`, :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

**Breaking Changes**
    * The ``DataColumn.to_pandas`` method was renamed to ``DataColumn.to_series``.
    * The ``DataTable.to_pandas`` method was renamed to ``DataTable.to_dataframe``.
    * ``copy`` is no longer a parameter of ``DataTable.to_dataframe`` or ``DataColumn.to_series``.

**v0.0.4 October 21, 2020**
    * Enhancements
        * Add optional ``include`` parameter for ``DataTable.describe()`` to filter results (:pr:`228`)
        * Add ``make_index`` parameter to ``DataTable.__init__`` to enable optional creation of a new index column (:pr:`238`)
        * Add support for setting ranking order on columns with Ordinal logical type (:pr:`240`)
        * Add ``list_semantic_tags`` function and CLI to get dataframe of woodwork semantic_tags (:pr:`244`)
        * Add support for numeric time index on DataTable (:pr:`267`)
        * Add pop method to DataTable (:pr:`289`)
        * Add entry point to setup.py to run CLI commands (:pr:`285`)
    * Fixes
        * Allow numeric datetime time indices (:pr:`282`)
    * Changes
        * Remove redundant methods ``DataTable.select_ltypes`` and ``DataTable.select_semantic_tags`` (:pr:`239`)
        * Make results of ``get_mutual_information`` more clear by sorting and removing self calculation (:pr:`247`)
        * Lower minimum scikit-learn version to 0.21.3 (:pr:`297`)
    * Documentation Changes
        * Add guide for ``dt.describe`` and ``dt.get_mutual_information`` (:pr:`245`)
        * Update README.md with documentation link (:pr:`261`)
        * Add footer to doc pages with Alteryx Open Source (:pr:`258`)
        * Add types and tags one-sentence definitions to Understanding Types and Tags guide (:pr:`271`)
        * Add issue and pull request templates (:pr:`280`, :pr:`284`)
    * Testing Changes
        * Add automated process to check latest dependencies. (:pr:`268`)
        * Add test for setting a time index with specified string logical type (:pr:`279`)

    Thanks to the following people for contributing to this release:
    :user:`ctduffy`, :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

**v0.0.3 October 9, 2020**
    * Enhancements
        * Implement setitem on DataTable to create/overwrite an existing DataColumn (:pr:`165`)
        * Add ``to_pandas`` method to DataColumn to access the underlying series (:pr:`169`)
        * Add list_logical_types function and CLI to get dataframe of woodwork LogicalTypes (:pr:`172`)
        * Add ``describe`` method to DataTable to generate statistics for the underlying data (:pr:`181`)
        * Add optional ``return_dataframe`` parameter to ``load_retail`` to return either DataFrame or DataTable (:pr:`189`)
        * Add ``get_mutual_information`` method to DataTable to generate mutual information between columns (:pr:`203`)
        * Add ``read_csv`` function to create DataTable directly from CSV file (:pr:`222`)
    * Fixes
        * Fix bug causing incorrect values for quartiles in ``DataTable.describe`` method (:pr:`187`)
        * Fix bug in ``DataTable.describe`` that could cause an error if certain semantic tags were applied improperly (:pr:`190`)
        * Fix bug with instantiated LogicalTypes breaking when used with issubclass (:pr:`231`)
    * Changes
        * Remove unnecessary ``add_standard_tags`` attribute from DataTable (:pr:`171`)
        * Remove standard tags from index column and do not return stats for index column from ``DataTable.describe`` (:pr:`196`)
        * Update ``DataColumn.set_semantic_tags`` and ``DataColumn.add_semantic_tags`` to return new objects (:pr:`205`)
        * Update various DataTable methods to return new objects rather than modifying in place (:pr:`210`)
        * Move datetime_format to Datetime LogicalType (:pr:`216`)
        * Do not calculate mutual info with index column in ``DataTable.get_mutual_information`` (:pr:`221`)
        * Move setting of underlying physical types from DataTable to DataColumn (:pr:`233`)
    * Documentation Changes
        * Remove unused code from sphinx conf.py, update with Github URL(:pr:`160`, :pr:`163`)
        * Update README and docs with new Woodwork logo, with better code snippets (:pr:`161`, :pr:`159`)
        * Add DataTable and DataColumn to API Reference (:pr:`162`)
        * Add docstrings to LogicalType classes (:pr:`168`)
        * Add Woodwork image to index, clear outputs of Jupyter notebook in docs (:pr:`173`)
        * Update contributing.md, release.md with all instructions (:pr:`176`)
        * Add section for setting index and time index to start notebook (:pr:`179`)
        * Rename changelog to Release Notes (:pr:`193`)
        * Add section for standard tags to start notebook (:pr:`188`)
        * Add Understanding Types and Tags user guide (:pr:`201`)
        * Add missing docstring to ``list_logical_types`` (:pr:`202`)
        * Add Woodwork Global Configuration Options guide (:pr:`215`)
    * Testing Changes
        * Add tests that confirm dtypes are as expected after DataTable init (:pr:`152`)
        * Remove unused ``none_df`` test fixture (:pr:`224`)
        * Add test for ``LogicalType.__str__`` method (:pr:`225`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

**v0.0.2 September 28, 2020**
    * Fixes
        * Fix formatting issue when printing global config variables (:pr:`138`)
    * Changes
        * Change add_standard_tags to use_standard_Tags to better describe behavior (:pr:`149`)
        * Change access of underlying dataframe to be through ``to_pandas`` with ._dataframe field on class (:pr:`146`)
        * Remove ``replace_none`` parameter to DataTables (:pr:`146`)
    * Documentation Changes
        * Add working code example to README and create Using Woodwork page (:pr:`103`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

**v0.1.0 September 24, 2020**
    * Add ``natural_language_threshold`` global config option used for Categorical/NaturalLanguage type inference (:pr:`135`)
    * Add global config options and add ``datetime_format`` option for type inference (:pr:`134`)
    * Fix bug with Integer and WholeNumber inference in column with ``pd.NA`` values (:pr:`133`)
    * Add ``DataTable.ltypes`` property to return series of logical types (:pr:`131`)
    * Add ability to create new datatable from specified columns with ``dt[[columns]]`` (:pr:`127`)
    * Handle setting and tagging of index and time index columns (:pr:`125`)
    * Add combined tag and ltype selection (:pr:`124`)
    * Add changelog, and update changelog check to CI (:pr:`123`)
    * Implement ``reset_semantic_tags`` (:pr:`118`)
    * Implement DataTable getitem (:pr:`119`)
    * Add ``remove_semantic_tags`` method (:pr:`117`)
    * Add semantic tag selection (:pr:`106`)
    * Add github action, rename to woodwork (:pr:`113`)
    * Add license to setup.py (:pr:`112`)
    * Reset semantic tags on logical type change (:pr:`107`)
    * Add standard numeric and category tags (:pr:`100`)
    * Change ``semantic_types`` to ``semantic_tags``, a set of strings (:pr:`100`)
    * Update dataframe dtypes based on logical types (:pr:`94`)
    * Add ``select_logical_types`` to DataTable (:pr:`96`)
    * Add pygments to dev-requirements.txt (:pr:`97`)
    * Add replacing None with np.nan in DataTable init (:pr:`87`)
    * Refactor DataColumn to make ``semantic_types`` and ``logical_type`` private (:pr:`86`)
    * Add pandas_dtype to each Logical Type, and remove dtype attribute on DataColumn (:pr:`85`)
    * Add set_semantic_types methods on both DataTable and DataColumn (:pr:`75`)
    * Support passing camel case or snake case strings for setting logical types (:pr:`74`)
    * Improve flexibility when setting semantic types (:pr:`72`)
    * Add Whole Number Inference of Logical Types (:pr:`66`)
    * Add ``dtypes`` property to DataTables and ``repr`` for DataColumn (:pr:`61`)
    * Allow specification of semantic types during DataTable creation (:pr:`69`)
    * Implements ``set_logical_types`` on DataTable (:pr:`65`)
    * Add init files to tests to fix code coverage (:pr:`60`)
    * Add AutoAssign bot (:pr:`59`)
    * Add logical types validation in DataTables (:pr:`49`)
    * Fix working_directory in CI (:pr:`57`)
    * Add ``infer_logical_types`` for DataColumn (:pr:`45`)
    * Fix ReadME library name, and code coverage badge (:pr:`56`, :pr:`56`)
    * Add code coverage (:pr:`51`)
    * Improve and refactor the validation checks during initialization of a DataTable (:pr:`40`)
    * Add dataframe attribute to DataTable (:pr:`39`)
    * Update ReadME with minor usage details (:pr:`37`)
    * Add License (:pr:`34`)
    * Rename from datatables to datatables (:pr:`4`)
    * Add Logical Types, DataTable, DataColumn (:pr:`3`)
    * Add Makefile, setup.py, requirements.txt (:pr:`2`)
    * Initial Release (:pr:`1`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

.. command
.. git log --pretty=oneline --abbrev-commit
