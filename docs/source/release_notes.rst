.. _release_notes:

Release Notes
-------------

Future Release
==============
    * Enhancements
        * Improved inference for numeric logical types to handle incoming ``object`` dtype data (:pr:`1645`)
        * Updated datetime format inference to handle years represented by 2 digits (:pr:`1632`)
        * Updated ``dependence_dict`` to handle boolean columns (:pr:`1652`)
    * Fixes
    * Changes
        * Pin for jupyter-client to 7.4.9 for documentation (:pr:`1624`)
        * Remove jupyter-client documentation requirement (:pr:`1627`)
    * Documentation Changes
    * Testing Changes
        * Add ruff for linting and replace isort/flake8 (:pr:`1614`)
        * Specify black and ruff config arguments (:pr:`1620`)
        * Add codecov token for unit tests workflow (:pr:`1630`)
        * Add GitHub Actions cache to speed up workflows (:pr:`1631`)
        * Add pull request check for linked issues to CI workflow (:pr:`1633`, :pr:`1636`)
        * Run lint fix on latest dependency update pull requests (:pr:`1640`, :pr:`1641`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`bchen1116`

v0.21.2 January 11, 2023
========================
    * Changes
        * Bump scipy and scikit-learn min versions for compatibility with numpy 1.24.0 (:pr:`1606`)
        * Add ``is_natural_language`` method to ``ColumnSchema`` object (:pr:`1610`)
        * Changed the ``transform`` function for the ``Boolean`` logical type to improve runtime (:pr:`1612`)

    Thanks to the following people for contributing to this release:
    :user:`ParthivNaresh`, :user:`sbadithe`, :user:`thehomebrewnerd`


v0.21.1 December 16, 2022
=========================
    * Fixes
        * Fix importlib DeprecationWarning in ``inference_functions.py`` (:pr:`1584`)
        * Schema now maintains column order after renaming a column (:pr:`1594`)
        * Fixed logic to not set config during boolean transform (:pr:`1601`)
    * Changes
        * Rename ``backup_dtype`` to ``pyspark_dtype`` (:pr:`1593`)
        * Removed inference for ``["0", "1"]``, ``["0.0", "1.0"]``, and ``[0, 1]`` as Boolean logical types, but maintained forced inference of such values (:pr:`1600`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`sbadithe`

v0.21.0 December 1, 2022
========================
    * Enhancements
        * Improved ``Boolean`` and ``BooleanNullable`` inference to detect common string representations of boolean values (:pr:`1549`)
        * Added the ``get_outliers`` and ``medcouple_dict`` functions to ``WoodworkColumnAccessor`` so that the medcouple statistic can be used for outlier detection (:pr:`1547`)
    * Fixes
        * Resolved FutureWarning in ``_get_box_plot_info_for_column`` (:pr:`1563`)
        * Fixed error message in validate method in logical_types.py (:pr:`1565`)
        * Fixed ``IntegerNullable`` inference by checking values are within valid ``Int64`` bounds (:pr:`1572`)
        * Update demo dataset links to point to new endpoint (:pr:`1570`)
        * Fix DivisionByZero error in ``type_system.py`` (:pr:`1571`)
        * Fix Categorical dtype inference for ``PostalCode`` logical type (:pr:`1574`)
        * Fixed issue where forcing a ``Boolean`` logical type on a column of 0.0s and 1.0s caused incorrect transformation (:pr:`1576`)
    * Changes
        * Unpin dask dependency (:pr:`1561`)
        * Changed the sampling strategy for type inference from ``head`` to random (:pr:`1566`)
    * Documentation Changes
        * Updated documentation to include the ``get_outliers`` and ``medcouple_dict`` (:pr:`1547`)
    * Testing Changes
        * Run looking glass performance tests on merge (:pr:`1567`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`gsheni`, :user:`ParthivNaresh`, :user:`sbadithe`, :user:`simha104`

Breaking Changes
++++++++++++++++
    * :pr:`1549` will automatically infer more values as ``Boolean`` or ``BooleanNullable``, including, but not limited to, ``[0, 1]``, ``['yes', 'no']``, and ``["True", "False"]``.

v0.20.0 October 31, 2022
========================
    * Enhancements
        * Replace use of deprecated ``append`` method for dataframes and series with ``concat`` method (:pr:`1533`)
    * Fixes
        * Fixed bug relating to ``dependence`` calculations to ensure columns exist in dataframe (:pr:`1534`)
        * Small typo fix in ``select`` docstring (:pr:`1544`)
        * Fix TypeValidationError message (:pr:`1557`)
        * Set dask version below 2022.10.1 (:pr:`1558`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`sbadithe`

v0.19.0 September 27, 2022
==========================
    * Enhancements
        * Added ``Spearman Correlation`` to options for dependence calculations (:pr:`1523`)
        * Added ``ignore_zeros`` as an argument for ``box_plot_dict`` to allow for calculations of outliers without 0 values (:pr:`1524`)
        * Added ``target_col`` argument to ``dependence`` and ``dependence_dict`` to calculate correlations between features and target_col (:pr:`1531`)
    * Fixes
        * Fix datetime pivot point to be set at current year + 10 rather than the default for two-digit years when ``datetime_format`` provided (:pr:`1512`)
    * Changes
        * Added ``ignore_columns`` as an argument when initializing a dataframe (:pr:`1504`)
        * Remove ``dask[dataframe]`` version restriction (:pr:`1527`)
    * Testing Changes
        * Add kickoff for create conda forge pull request from release (:pr:`1515`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`gsheni`, :user:`ParthivNaresh`, :user:`thehomebrewnerd`

v0.18.0 August 31, 2022
=======================
    * Enhancements
        * Updated ``dependence_dict`` and ``mutual_information`` to drop to drop Categorical columns with a large number of unique values during mutual information calculation, non-dask only. (:pr:`1501`)
    * Fixes
        * Fix applying LatLong.transform to empty dask data (:pr:`1507`)
    * Changes
        * Transition from setup.cfg to pyproject.toml (:pr:`1506`,:pr:`1508`)
        * Added a check to see if a series dtype has changed prior to using ``_replace_nans`` (:pr:`1502`)
    * Testing Changes
        * Update development requirements and use latest for documentation (:pr:`1499`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`ParthivNaresh`, :user:`rwedge`

v0.17.2 August 5, 2022
======================
    * Fixes
        * Updated concat_columns to work with dataframes with mismatched indices or different shapes (:pr:`1485`)
    * Documentation Changes
        * Add instructions to add new users to woodwork feedstock (:pr:`1483`)
    * Testing Changes
        * Add create feedstock PR workflow (:pr:`1489`)

    Thanks to the following people for contributing to this release:
    :user:`chukarsten`, :user:`cmancuso`, :user:`gsheni`,

v0.17.1 July 29, 2022
=====================
    * Testing Changes
       * Allow for manual kickoff for minimum dependency checker (:pr:`1476`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`gsheni`

v0.17.0 July 14, 2022
=====================
    .. warning::
            This release of Woodwork will not support Python 3.7

    * Enhancements
        * Added ability to null invalid values for ``Double`` logical type (:pr:`1449`)
        * Added ability to null invalid values for ``BooleanNullable`` logical type (:pr:`1455`)
        * Added ability to null invalid values for ``IntegerNullable`` logical type (:pr:`1456`)
        * Added ability to null invalid values for ``EmailAddress`` logical type (:pr:`1457`)
        * Added ability to null invalid values for ``URL`` logical type (:pr:`1459`)
        * Added ability to null invalid values for ``PhoneNumber`` logical type (:pr:`1460`)
        * Added ability to null invalid values for ``AgeFractional`` and ``AgeNullable`` logical types (:pr:`1462`)
        * Added ability to null invalid values for ``LatLong`` logical type (:pr:`1465`)
        * Added ability to null invalid values for ``PostalCode`` logical type (US only) (:pr:`1467`)
        * Added smarter inference for ``IntegerNullable`` and ``BooleanNullable`` types (:pr:`1458`)
    * Fixes
        * Fixed inference of all null string values as ``Unknown`` instead of ``Datetime`` (:pr:`1458`)
    * Changes
        * Set the minimum acceptable version of pandas to ``1.4.0`` for woodwork and ``1.4.3`` for spark add-on (:pr:`1461`)
        * Dropped support for Python 3.7 (:pr:`1461`)
        * Add pre-commit hooks for linting (:pr:`1470`)

   Thanks to the following people for contributing to this release:
   :user:`gsheni`, :user:`jeff-hernandez`, :user:`ParthivNaresh`

v0.16.4 Jun 23, 2022
====================
    * Fixes
        * Fix concatenation of invalid logical type values (:pr:`1437`)
        * Fix validation for numeric postal codes (:pr:`1439`)
    * Changes
        * Restrict pyspark below 3.3.0 (:pr:`1450`)
    * Documentation Changes
        * Add slack icon to footer in docs (:pr:`1432`)
        * Update contributing.md to add pandoc (:pr:`1443`)
    * Testing Changes
        * Use codecov action v3 (:pr:`1422`)
        * Added tests to test minimum dependencies of minimum dependencies (:pr:`1440`)
        * Add workflow to kickoff EvalML unit tests on commit to main (:pr:`1424`, :pr:`1426`)
        * Rename yml to yaml for GitHub Actions (:pr:`1428`, :pr:`1429`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`ParthivNaresh`

v0.16.3 May 4, 2022
===================
    * Fixes
        * Fixed ``col_is_datetime`` inference function to not infer numeric dtypes as datetime (:pr:`1413`)
    * Changes
        * Delete setup.py, MANIFEST.in and move configuration to pyproject.toml (:pr:`1409`)
    * Documentation Changes
        * Update slack invite link to new (:pr:`1406`, :pr:`1407`, :pr:`1408`)
    * Testing Changes
        * Add workflow to kickoff Featuretools unit tests with Woodwork main (:pr:`1400`)
        * Add workflow for testing Woodwork without test dependencies (:pr:`1414`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`gsheni`, :user:`ParthivNaresh`

v0.16.2 Apr 25, 2022
====================
    * Fixes
        * Fixed import issues regarding ``pyarrow`` and made ``python-dateutil>=2.8.1`` a required dependency (:pr:`1397`)

    Thanks to the following people for contributing to this release:
    :user:`ParthivNaresh`

v0.16.1 Apr 25, 2022
====================
    * Fixes
        * Reverting ``string[pyarrow]`` until fix can be found for pandas issue (:pr:`1391`)

    Thanks to the following people for contributing to this release:
    :user:`ParthivNaresh`

v0.16.0 Apr 21, 2022
====================
    * Enhancements
        * Added the ability to provide a callback function to ``TableAccessor.describe()`` to get intermediate results (:pr:`1387`)
        * Add ``pearson_correlation`` and ``dependence`` methods to TableAccessor (:pr:`1265`)
        * Uses ``string[pyarrow]`` instead of ``string`` dtype to save memory (:pr:`1360`)
        * Added a better error message when dataframe and schema have different columns (:pr:`1366`)
        * Stores timezones in Datetime logical type (:pr:`1376`)
        * Added type inference for phone numbers (:pr:`1357`)
        * Added type inference for zip code (:pr:`1378`)
    * Fixes
        * Cap pandas at 1.4.1 (:pr:`1373`)
    * Changes
        * Change underlying logic of ``TableAccessor.mutual_information`` (:pr:`1265`)
        * Added ``from_disk`` as a convenience function to deserialize a WW table (:pr:`1363`)
        * Allow attr version in setup.cfg (:pr:`1361`)
        * Raise error if files already exist during serialization (:pr:`1356`)
        * Improve exception handling in ``col_is_datetime`` (:pr:`1365`)
        * Store typing info in parquet file header during serialization (:pr:`1377`)
    * Documentation Changes
        * Upgrade nbconvert and remove jinja2 dependency (:pr:`1362`)
        * Add M1 installation instructions to docs and contributing guide (:pr:`1367`)
        * Update README text to Alteryx (:pr:`1381`, :pr:`1382`)
    * Testing Changes
        * Separate testing matrix to speed up GitHub Actions Linux tests for latest dependencies :pr:`1380`

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`ParthivNaresh`, :user:`rwedge`, :user:`thehomebrewnerd`

v0.15.0 Mar 24, 2022
====================
    * Enhancements
        * Added ``CurrencyCode`` to logical types (:pr:`1348`)
        * Added Datetime Frequency Inference V2 (:pr:`1281`)
    * Fixes
        * Updated ``__str__`` output for ``Ordinal`` logical types (:pr:`1340`)
    * Changes
        * Updated lint check to only run on Python 3.10 (:pr:`1345`)
        * Transition to use pyproject.toml and setup.cfg (moving away from setup.py) (:pr:`1346`)
    * Documentation Changes
        * Update release.md with correct version updating info (:pr:`1358`)
    * Testing Changes
        * Updated scheduled workflows to only run on Alteryx owned repos (:pr:`1351`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`dvreed77`, :user:`jeff-hernandez`, :user:`ParthivNaresh`, :user:`thehomebrewnerd`

v0.14.0 Mar 15, 2022
====================
    * Fixes
        * Preserve custom semantic tags when changing column logical type (:pr:`1300`)
    * Changes
        * Calculate nunique for Unknown columns in `_get_describe_dict`  (:pr:`1322`)
        * Refactor serialization and deserialization for improved modularity (:pr:`1325`)
        * Replace Koalas with the pandas API on Spark (:pr:`1331`)
    * Documentation Changes
        * Update copy and paste button to remove syntax signs (:pr:`1313`)
        * Move LatLong and Ordinal logical type validation logic to LogicalType.validate methods (:pr:`1315`)
        * Add backport release support (:pr:`1321`)
        * Add `get_subset_schema` to API reference (:pr:`1335`)
    * Testing Changes
        * Upgrade moto dependency requirement (:pr:`1327`, :pr:`1332`)
        * Add Python 3.10 support (:pr:`1316`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`mingdavidqi`

Breaking Changes
++++++++++++++++
    * :pr:`1325`: The following serialization functions have been removed from the API:
      ``woodwork.serialize.write_dataframe``, ``woodwork.serialize.write_typing_info`` and
      ``woodwork.serialize.write_woodwork_table``. Also, the function ``woodwork.serialize.typing_info_to_dict``
      has been moved to ``woodwork.serializers.serializer_base.typing_info_to_dict``.

v0.13.0 Feb 16, 2022
====================
    .. warning::
        Woodwork may not support Python 3.7 in next non-bugfix release.

    * Enhancements
        * Add validation to EmailAddress logical type (:pr:`1247`)
        * Add validation to URL logical type (:pr:`1285`)
        * Add validation to Age, AgeFractional, and AgeNullable logical types (:pr:`1289`)
    * Fixes
        * Check range length in table stats without producing overflow error (:pr:`1287`)
        * Fixes issue with initializing Woodwork Series with LatLong values (:pr:`1299`)
    * Changes
        * Remove framework for unused ``woodwork`` CLI (:pr:`1288`)
        * Add back support for Python 3.7 (:pr:`1292`)
        * Nested statistical utility functions into directory (:pr:`1295`)
    * Documentation Changes
        * Updating contributing doc with PATH and JAVA_HOME instructions (:pr:`1273`)
        * Better install page with new Sphinx extensions for copying and in-line tabs (:pr:`1280`, :pr:`1282`)
        * Update README.md with Alteryx link (:pr:`1291`)
    * Testing Changes
        * Replace mock with unittest.mock (:pr:`1304`)

    Thanks to the following people for contributing to this release:
    :user:`dvreed77`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`


v0.12.0 Jan 27, 2022
====================
    * Enhancements
        * Add Slack link to GitHub issue creation templates (:pr:`1242`)
    * Fixes
        * Fixed issue with tuples being incorrectly inferred as EmailAddress (:pr:`1253`)
        * Set high and low bounds to the max and min values if no outliers are present in ``box_plot_dict`` (:pr:`1269`)
    * Changes
        * Prevent setting index that contains null values (:pr:`1239`)
        * Allow tuple NaN LatLong values (:pr:`1255`)
        * Update ipython to 7.31.1 (:pr:`1258`)
        * Temporarily restrict pandas and koalas max versions (:pr:`1261`)
        * Update to drop Python 3.7 support and add support for pandas version 1.4.0 (:pr:`1264`)
    * Testing Changes
        * Change auto approve workflow to use PR number (:pr:`1240`, :pr:`1241`)
        * Update auto approve workflow to delete branch and change on trigger (:pr:`1251`)
        * Fix permissions issue with S3 deserialization test (:pr:`1238`)

    Thanks to the following people for contributing to this release:
    :user:`dvreed77`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`

v0.11.2 Jan 28, 2022
====================
    * Fixes
        * Set high and low bounds to the max and min values if no outliers are present in ``box_plot_dict`` (backport of :pr:`1269`)

    Thanks to the following people for contributing to this release:
    :user:`tamargrey`

Note
++++
    * The pandas version for Koalas has been restricted, and a change was made to a pandas ``replace`` call to account for
      the recent pandas 1.4.0 release.

v0.11.1 Jan 4, 2022
===================
    * Changes
        * Update inference process to only check for NaturalLanguage if no other type matches are found first (:pr:`1234`)
    * Documentation Changes
        * Updating contributing doc with Spark installation instructions (:pr:`1232`)
    * Testing Changes
        * Enable auto-merge for minimum and latest dependency merge requests (:pr:`1228`, :pr:`1230`, :pr:`1233`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`thehomebrewnerd`, :user:`willsmithorg`

v0.11.0 Dec 22, 2021
====================
    * Enhancements
        * Add type inference for natural language (:pr:`1210`)
    * Changes
        * Make public method ``get_subset_schema`` (:pr:`1218`)

   Thanks to the following people for contributing to this release:
   :user:`jeff-hernandez`, :user:`thehomebrewnerd`, :user:`tuethan1999`

v0.10.0 Nov 30, 2021
====================
    * Enhancements
        * Allow frequency inference on temporal (Datetime, Timedelta) columns of Woodwork DataFrame (:pr:`1202`)
        * Update ``describe_dict`` to compute ``top_values`` for double columns that contain only integer values (:pr:`1206`)
    * Changes
        * Return histogram bins as a list of floats instead of a ``pandas.Interval`` object (:pr:`1207`)

    Thanks to the following people for contributing to this release:
    :user:`tamargrey`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
    * :pr:``1207``: The behavior of ``describe_dict`` has changed when using
      ``extra_stats=True``. Previously, the histogram bins were returned as
      ``pandas.Interval`` objects. This has been updated so that the histogram
      bins are now represented as a two-element list of floats with the first element
      being the left edge of the bin and the second element being the right edge.

v0.9.1 Nov 19, 2021
===================
    * Fixes
        * Fix bug that causes ``mutual_information`` to fail with certain index types (:pr:`1199`)
    * Changes
        * Update pip to 21.3.1 for test requirements (:pr:`1196`)
    * Documentation Changes
        * Update install page with updated minimum optional dependencies (:pr:`1193`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`thehomebrewnerd`

v0.9.0 Nov 11, 2021
===================
    * Enhancements
        * Added ``read_file`` parameter for replacing empty string values with NaN values (:pr:`1161`)
    * Fixes
        * Set a maximum version for pyspark until we understand why :pr:`1169` failed (:pr:`1179`)
        * Require newer dask version (:pr:`1180`)
    * Changes
        * Make box plot low/high indices/values optional to return in ``box_plot_dict`` (:pr:`1184`)
    * Documentation Changes
        * Update docs dependencies (:pr:`1176`)
    * Testing Changes
        * Add black linting package and remove autopep8 (:pr:`1164`, :pr:`1183`)
        * Updated notebook standardizer to standardize python versions (:pr:`1166`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`davesque`, :user:`gsheni`,  :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`

v0.8.2 Oct 12, 2021
===================
    * Fixes
        * Fixed an issue when inferring the format of datetime strings with day of week or meridiem placeholders (:pr:`1158`)
        * Implements change in ``Datetime.transform`` to prevent initialization failure in some cases (:pr:`1162`)
    * Testing Changes
        * Update reviewers for minimum and latest dependency checkers (:pr:`1150`)
        * Added notebook standardizer to remove executed outputs (:pr:`1153`)

    Thanks to the following people for contributing to this release:
    :user:`bchen1116`, :user:`davesque`, :user:`jeff-hernandez`, :user:`thehomebrewnerd`

v0.8.1 Sep 16, 2021
===================
    * Changes
        * Update ``Datetime.transform`` to use default nrows value when calling ``_infer_datetime_format`` (:pr:`1137`)
    * Documentation Changes
        * Hide spark config in Using Dask and Koalas Guide (:pr:`1139`)

    Thanks to the following people for contributing to this release:
    :user:`jeff-hernandez`, :user:`simha104`, :user:`thehomebrewnerd`


v0.8.0 Sep 9, 2021
==================
    * Enhancements
        * Add support for automatically inferring the ``URL`` and ``IPAddress`` logical types (:pr:`1122`, :pr:`1124`)
        * Add ``get_valid_mi_columns`` method to list columns that have valid logical types for mutual information calculation (:pr:`1129`)
        * Add attribute to check if column has a nullable logical type (:pr:`1127`)
    * Changes
        * Update ``get_invalid_schema_message`` to improve performance (:pr:`1132`)
    * Documentation Changes
        * Fix typo in the "Get Started" documentation (:pr:`1126`)
        * Clean up the logical types guide (:pr:`1134`)

    Thanks to the following people for contributing to this release:
    :user:`ajaypallekonda`, :user:`davesque`, :user:`jeff-hernandez`, :user:`thehomebrewnerd`

v0.7.1 Aug 25, 2021
===================
    * Fixes
        * Validate schema's index if being used in partial schema init (:pr:`1115`)
        * Allow falsy index, time index, and name values to be set along with partial schema at init (:pr:`1115`)

    Thanks to the following people for contributing to this release:
    :user:`tamargrey`

v0.7.0 Aug 25, 2021
===================
    * Enhancements
        * Add ``'passthrough'`` and ``'ignore'`` to tags in ``list_semantic_tags`` (:pr:`1094`)
        * Add initialize with partial table schema  (:pr:`1100`)
        * Apply ordering specified by the ``Ordinal`` logical type to underlying series (:pr:`1097`)
        * Add ``AgeFractional`` logical type (:pr:`1112`)

    Thanks to the following people for contributing to this release:
    :user:`davesque`, :user:`jeff-hernandez`, :user:`tamargrey`, :user:`tuethan1999`

Breaking Changes
++++++++++++++++
    * :pr:``1100``: The behavior for ``init`` has changed. A full schema is a
      schema that contains all of the columns of the dataframe it describes
      whereas a partial schema only contains a subset. A full schema will also
      require that the schema is valid without having to make any changes to
      the DataFrame. Before, only a full schema was permitted by the ``init``
      method so passing a partial schema would error. Additionally, any
      parameters like ``logical_types`` would be ignored if passing in a schema.
      Now, passing a partial schema to the ``init`` method calls the
      ``init_with_partial_schema`` method instead of throwing an error.
      Information from keyword arguments will override information from the
      partial schema. For example, if column ``a`` has the Integer Logical Type
      in the partial schema, it's possible to use the ``logical_type`` argument
      to reinfer it's logical type by passing ``{'a': None}`` or force a type by
      passing in ``{'a': Double}``. These changes mean that Woodwork init is less
      restrictive. If no type inference takes place and no changes are required
      of the DataFrame at initialization, ``init_with_full_schema`` should be
      used instead of ``init``. ``init_with_full_schema`` maintains the same
      functionality as when a schema was passed to the old ``init``.

v0.6.0 Aug 4, 2021
==================
    * Fixes
        * Fix bug in ``_infer_datetime_format`` with all ``np.nan`` input (:pr:`1089`)
    * Changes
        * The criteria for categorical type inference have changed (:pr:`1065`)
        * The meaning of both the ``categorical_threshold`` and
          ``numeric_categorical_threshold`` settings have changed (:pr:`1065`)
        * Make sampling for type inference more consistent (:pr:`1083`)
        * Accessor logic checking if Woodwork has been initialized moved to decorator (:pr:`1093`)
    * Documentation Changes
        * Fix some release notes that ended up under the wrong release (:pr:`1082`)
        * Add BooleanNullable and IntegerNullable types to the docs (:pr:`1085`)
        * Add guide for saving and loading Woodwork DataFrames (:pr:`1066`)
        * Add in-depth guide on logical types and semantic tags (:pr:`1086`)
    * Testing Changes
        * Add additional reviewers to minimum and latest dependency checkers (:pr:`1070`, :pr:`1073`, :pr:`1077`)
        * Update the sample_df fixture to have more logical_type coverage (:pr:`1058`)

    Thanks to the following people for contributing to this release:
    :user:`davesque`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`tuethan1999`

Breaking Changes
++++++++++++++++
    * :pr:`1065`: The criteria for categorical type inference have changed.
      Relatedly, the meaning of both the ``categorical_threshold`` and
      ``numeric_categorical_threshold`` settings have changed.  Now, a
      categorical match is signaled when a series either has the "categorical"
      pandas dtype *or* if the ratio of unique value count (nan excluded) and
      total value count (nan also excluded) is below or equal to some fraction.
      The value used for this fraction is set by the ``categorical_threshold``
      setting which now has a default value of ``0.2``.  If a fraction is set
      for the ``numeric_categorical_threshold`` setting, then series with
      either a float or integer dtype may be inferred as categorical by
      applying the same logic described above with the
      ``numeric_categorical_threshold`` fraction.  Otherwise, the
      ``numeric_categorical_threshold`` setting defaults to ``None`` which
      indicates that series with a numerical type should not be inferred as
      categorical.  Users who have overridden either the
      ``categorical_threshold`` or ``numeric_categorical_threshold`` settings
      will need to adjust their settings accordingly.
    * :pr:`1083`: The process of sampling series for logical type inference was
      updated to be more consistent.  Before, initial sampling for inference
      differed depending on collection type (pandas, dask, or koalas).  Also,
      further randomized subsampling was performed in some cases during
      categorical inference and in every case during email inference regardless
      of collection type.  Overall, the way sampling was done was inconsistent
      and unpredictable.  Now, the first 100,000 records of a column are
      sampled for logical type inference regardless of collection type although
      only records from the first partition of a dask dataset will be used.
      Subsampling performed by the inference functions of individual types has
      been removed.  The effect of these changes is that inferred types may now
      be different although in many cases they will be more correct.

v0.5.1 Jul 22, 2021
===================
    * Enhancements
        * Store inferred datetime format on Datetime logical type instance (:pr:`1025`)
        * Add support for automatically inferring the ``EmailAddress`` logical type (:pr:`1047`)
        * Add feature origin attribute to schema (:pr:`1056`)
        * Add ability to calculate outliers and the statistical info required for box and whisker plots to ``WoodworkColumnAccessor`` (:pr:`1048`)
        * Add ability to change config settings in a with block with ``ww.config.with_options`` (:pr:`1062`)
    * Fixes
        * Raises warning and removes tags when user adds a column with index tags to DataFrame (:pr:`1035`)
    * Changes
        * Entirely null columns are now inferred as the Unknown logical type (:pr:`1043`)
        * Add helper functions that check for whether an object is a koalas/dask series or dataframe (:pr:`1055`)
        * ``TableAccessor.select`` method will now maintain dataframe column ordering in TableSchema columns (:pr:`1052`)
    * Documentation Changes
        * Add supported types to metadata docstring (:pr:`1049`)

    Thanks to the following people for contributing to this release:
    :user:`davesque`, :user:`frances-h`, :user:`jeff-hernandez`, :user:`simha104`, :user:`tamargrey`, :user:`thehomebrewnerd`

v0.5.0 Jul 7, 2021
==================
    * Enhancements
        * Add support for numpy array inputs to Woodwork (:pr:`1023`)
        * Add support for pandas.api.extensions.ExtensionArray inputs to Woodwork (:pr:`1026`)
    * Fixes
        * Add input validation to ww.init_series (:pr:`1015`)
    * Changes
        * Remove lines in ``LogicalType.transform`` that raise error if dtype conflicts (:pr:`1012`)
        * Add ``infer_datetime_format`` param to speed up ``to_datetime`` calls (:pr:`1016`)
        * The default logical type is now the ``Unknown`` type instead of the ``NaturalLanguage`` type (:pr:`992`)
        * Add pandas 1.3.0 compatibility (:pr:`987`)

    Thanks to the following people for contributing to this release:
    :user:`jeff-hernandez`, :user:`simha104`, :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`tuethan1999`

Breaking Changes
++++++++++++++++
    * The default logical type is now the ``Unknown`` type instead of the ``NaturalLanguage`` type.
      The global config ``natural_language_threshold`` has been renamed to ``categorical_threshold``.

v0.4.2 Jun 23, 2021
===================
    * Enhancements
        * Pass additional progress information in callback functions (:pr:`979`)
        * Add the ability to generate optional extra stats with ``DataFrame.ww.describe_dict`` (:pr:`988`)
        * Add option to read and write orc files (:pr:`997`)
        * Retain schema when calling ``series.ww.to_frame()`` (:pr:`1004`)
    * Fixes
        * Raise type conversion error in ``Datetime`` logical type (:pr:`1001`)
        * Try collections.abc to avoid deprecation warning (:pr:`1010`)
    * Changes
        * Remove ``make_index`` parameter from ``DataFrame.ww.init`` (:pr:`1000`)
        * Remove version restriction for dask requirements (:pr:`998`)
    * Documentation Changes
        * Add instructions for installing the update checker (:pr:`993`)
        * Disable pdf format with documentation build (:pr:`1002`)
        * Silence deprecation warnings in documentation build (:pr:`1008`)
        * Temporarily remove update checker to fix docs warnings (:pr:`1011`)
    * Testing Changes
        * Add env setting to update checker (:pr:`978`, :pr:`994`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`tamargrey`, :user:`thehomebrewnerd`, :user:`tuethan1999`

Breaking Changes
++++++++++++++++
    * Progress callback functions parameters have changed and progress is now being reported in the units
      specified by the unit of measurement parameter instead of percentage of total. Progress callback
      functions now are expected to accept the following five parameters:

        * progress increment since last call
        * progress units complete so far
        * total units to complete
        * the progress unit of measurement
        * time elapsed since start of calculation
    * ``DataFrame.ww.init`` no longer accepts the make_index parameter


v0.4.1 Jun 9, 2021
==================
    * Enhancements
        * Add ``concat_columns`` util function to concatenate multiple Woodwork objects into one, retaining typing information (:pr:`932`)
        * Add option to pass progress callback function to mutual information functions (:pr:`958`)
        * Add optional automatic update checker (:pr:`959`, :pr:`970`)
    * Fixes
        * Fix issue related to serialization/deserialization of data with whitespace and newline characters (:pr:`957`)
        * Update to allow initializing a ``ColumnSchema`` object with an ``Ordinal`` logical type without order values (:pr:`972`)
    * Changes
        * Change write_dataframe to only copy dataframe if it contains LatLong (:pr:`955`)
    * Testing Changes
        * Fix bug in ``test_list_logical_types_default`` (:pr:`954`)
        * Update minimum unit tests to run on all pull requests (:pr:`952`)
        * Pass token to authorize uploading of codecov reports (:pr:`969`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`


v0.4.0 May 26, 2021
===================
    * Enhancements
        * Add option to return ``TableSchema`` instead of ``DataFrame`` from table accessor ``select`` method (:pr:`916`)
        * Add option to read and write arrow/feather files (:pr:`948`)
        * Add dropping and renaming columns inplace (:pr:`920`)
        * Add option to pass progress callback function to mutual information functions (:pr:`943`)
    * Fixes
        * Fix bug when setting table name and metadata through accessor (:pr:`942`)
        * Fix bug in which the dtype of category values were not restored properly on deserialization (:pr:`949`)
    * Changes
        * Add logical type method to transform data (:pr:`915`)
    * Testing Changes
        * Update when minimum unit tests will run to include minimum text files (:pr:`917`)
        * Create separate workflows for each CI job (:pr:`919`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`thehomebrewnerd`, :user:`tuethan1999`

v0.3.1 May 12, 2021
===================
    .. warning::
        This Woodwork release uses a weak reference for maintaining a reference from the
        accessor to the DataFrame. Because of this, chaining a Woodwork call onto another
        call that creates a new DataFrame or Series object can be problematic.

        Instead of calling ``pd.DataFrame({'id':[1, 2, 3]}).ww.init()``, first store the DataFrame in a new
        variable and then initialize Woodwork:

        .. code-block:: python

            df = pd.DataFrame({'id':[1, 2, 3]})
            df.ww.init()


    * Enhancements
        * Add ``deep`` parameter to Woodwork Accessor and Schema equality checks (:pr:`889`)
        * Add support for reading from parquet files to ``woodwork.read_file`` (:pr:`909`)
    * Changes
        * Remove command line functions for list logical and semantic tags (:pr:`891`)
        * Keep index and time index tags for single column when selecting from a table (:pr:`888`)
        * Update accessors to store weak reference to data (:pr:`894`)
    * Documentation Changes
        * Update nbsphinx version to fix docs build issue (:pr:`911`, :pr:`913`)
    * Testing Changes
        * Use Minimum Dependency Generator GitHub Action and remove tools folder (:pr:`897`)
        * Move all latest and minimum dependencies into 1 folder (:pr:`912`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`tamargrey`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
    * The command line functions ``python -m woodwork list-logical-types`` and ``python -m woodwork list-semantic-tags``
      no longer exist. Please call the underlying Python functions ``ww.list_logical_types()`` and
      ``ww.list_semantic_tags()``.

v0.3.0 May 3, 2021
==================
    * Enhancements
        * Add ``is_schema_valid`` and ``get_invalid_schema_message`` functions for checking schema validity (:pr:`834`)
        * Add logical type for ``Age`` and ``AgeNullable`` (:pr:`849`)
        * Add logical type for ``Address`` (:pr:`858`)
        * Add generic ``to_disk`` function to save Woodwork schema and data (:pr:`872`)
        * Add generic ``read_file`` function to read file as Woodwork DataFrame (:pr:`878`)
    * Fixes
        * Raise error when a column is set as the index and time index (:pr:`859`)
        * Allow NaNs in index for schema validation check (:pr:`862`)
        * Fix bug where invalid casting to ``Boolean`` would not raise error (:pr:`863`)
    * Changes
        * Consistently use ``ColumnNotPresentError`` for mismatches between user input and dataframe/schema columns (:pr:`837`)
        * Raise custom ``WoodworkNotInitError`` when accessing Woodwork attributes before initialization (:pr:`838`)
        * Remove check requiring ``Ordinal`` instance for initializing a ``ColumnSchema`` object (:pr:`870`)
        * Increase koalas min version to 1.8.0 (:pr:`885`)
    * Documentation Changes
        * Improve formatting of release notes (:pr:`874`)
    * Testing Changes
        * Remove unnecessary argument in codecov upload job (:pr:`853`)
        * Change from GitHub Token to regenerated GitHub PAT dependency checkers (:pr:`855`)
        * Update README.md with non-nullable dtypes in code example (:pr:`856`)

    Thanks to the following people for contributing to this release:
    :user:`frances-h`, :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
    * Woodwork tables can no longer be saved using to disk ``df.ww.to_csv``, ``df.ww.to_pickle``, or
      ``df.ww.to_parquet``. Use ``df.ww.to_disk`` instead.
    * The ``read_csv`` function has been replaced by ``read_file``.


v0.2.0 Apr 20, 2021
===================
    .. warning::
        This Woodwork release does not support Python 3.6

    * Enhancements
        * Add validation control to WoodworkTableAccessor (:pr:`736`)
        * Store ``make_index`` value on WoodworkTableAccessor (:pr:`780`)
        * Add optional ``exclude`` parameter to WoodworkTableAccessor ``select`` method (:pr:`783`)
        * Add validation control to ``deserialize.read_woodwork_table`` and ``ww.read_csv`` (:pr:`788`)
        * Add ``WoodworkColumnAccessor.schema`` and handle copying column schema (:pr:`799`)
        * Allow initializing a ``WoodworkColumnAccessor`` with a ``ColumnSchema`` (:pr:`814`)
        * Add ``__repr__`` to ``ColumnSchema`` (:pr:`817`)
        * Add ``BooleanNullable`` and ``IntegerNullable`` logical types (:pr:`830`)
        * Add validation control to ``WoodworkColumnAccessor`` (:pr:`833`)
    * Changes
        * Rename ``FullName`` logical type to ``PersonFullName`` (:pr:`740`)
        * Rename ``ZIPCode`` logical type to ``PostalCode`` (:pr:`741`)
        * Fix issue with smart-open version 5.0.0 (:pr:`750`, :pr:`758`)
        * Update minimum scikit-learn version to 0.22 (:pr:`763`)
        * Drop support for Python version 3.6 (:pr:`768`)
        * Remove ``ColumnNameMismatchWarning`` (:pr:`777`)
        * ``get_column_dict`` does not use standard tags by default (:pr:`782`)
        * Make ``logical_type`` and ``name`` params to ``_get_column_dict`` optional (:pr:`786`)
        * Rename Schema object and files to match new table-column schema structure (:pr:`789`)
        * Store column typing information in a ``ColumnSchema`` object instead of a dictionary (:pr:`791`)
        * ``TableSchema`` does not use standard tags by default (:pr:`806`)
        * Store ``use_standard_tags`` on the ``ColumnSchema`` instead of the ``TableSchema`` (:pr:`809`)
        * Move functions in ``column_schema.py`` to be methods on ``ColumnSchema`` (:pr:`829`)
    * Documentation Changes
        * Update Pygments version requirement (:pr:`751`)
        * Update spark config for docs build (:pr:`787`, :pr:`801`, :pr:`810`)
    * Testing Changes
        * Add unit tests against minimum dependencies for python 3.6 on PRs and main (:pr:`743`, :pr:`753`, :pr:`763`)
        * Update spark config for test fixtures (:pr:`787`)
        * Separate latest unit tests into pandas, dask, koalas (:pr:`813`)
        * Update latest dependency checker to generate separate core, koalas, and dask dependencies (:pr:`815`, :pr:`825`)
        * Ignore latest dependency branch when checking for updates to the release notes (:pr:`827`)
        * Change from GitHub PAT to auto generated GitHub Token for dependency checker (:pr:`831`)
        * Expand ``ColumnSchema`` semantic tag testing coverage and null ``logical_type`` testing coverage (:pr:`832`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`rwedge`, :user:`tamargrey`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
    * The ``ZIPCode`` logical type has been renamed to ``PostalCode``
    * The ``FullName`` logical type has been renamed to ``PersonFullName``
    * The ``Schema`` object has been renamed to ``TableSchema``
    * With the ``ColumnSchema`` object, typing information for a column can no longer be accessed
      with ``df.ww.columns[col_name]['logical_type']``. Instead use ``df.ww.columns[col_name].logical_type``.
    * The ``Boolean`` and ``Integer`` logical types will no longer work with data that contains null
      values. The new ``BooleanNullable`` and ``IntegerNullable`` logical types should be used if
      null values are present.

v0.1.0 Mar 22, 2021
===================
    * Enhancements
        * Implement Schema and Accessor API (:pr:`497`)
        * Add Schema class that holds typing info (:pr:`499`)
        * Add WoodworkTableAccessor class that performs type inference and stores Schema (:pr:`514`)
        * Allow initializing Accessor schema with a valid Schema object (:pr:`522`)
        * Add ability to read in a csv and create a DataFrame with an initialized Woodwork Schema (:pr:`534`)
        * Add ability to call pandas methods from Accessor (:pr:`538`, :pr:`589`)
        * Add helpers for checking if a column is one of Boolean, Datetime, numeric, or categorical (:pr:`553`)
        * Add ability to load demo retail dataset with a Woodwork Accessor (:pr:`556`)
        * Add ``select`` to WoodworkTableAccessor (:pr:`548`)
        * Add ``mutual_information`` to WoodworkTableAccessor (:pr:`571`)
        * Add WoodworkColumnAccessor class (:pr:`562`)
        * Add semantic tag update methods to column accessor (:pr:`573`)
        * Add ``describe`` and ``describe_dict`` to WoodworkTableAccessor (:pr:`579`)
        * Add ``init_series`` util function for initializing a series with dtype change (:pr:`581`)
        * Add ``set_logical_type`` method to WoodworkColumnAccessor (:pr:`590`)
        * Add semantic tag update methods to table schema (:pr:`591`)
        * Add warning if additional parameters are passed along with schema (:pr:`593`)
        * Better warning when accessing column properties before init (:pr:`596`)
        * Update column accessor to work with LatLong columns (:pr:`598`)
        * Add ``set_index`` to WoodworkTableAccessor (:pr:`603`)
        * Implement ``loc`` and ``iloc`` for WoodworkColumnAccessor (:pr:`613`)
        * Add ``set_time_index`` to WoodworkTableAccessor (:pr:`612`)
        * Implement ``loc`` and ``iloc`` for WoodworkTableAccessor (:pr:`618`)
        * Allow updating logical types with ``set_types`` and make relevant DataFrame changes (:pr:`619`)
        * Allow serialization of WoodworkColumnAccessor to csv, pickle, and parquet (:pr:`624`)
        * Add DaskColumnAccessor (:pr:`625`)
        * Allow deserialization from csv, pickle, and parquet to Woodwork table (:pr:`626`)
        * Add ``value_counts`` to WoodworkTableAccessor (:pr:`632`)
        * Add KoalasColumnAccessor (:pr:`634`)
        * Add ``pop`` to WoodworkTableAccessor (:pr:`636`)
        * Add ``drop`` to WoodworkTableAccessor (:pr:`640`)
        * Add ``rename`` to WoodworkTableAccessor (:pr:`646`)
        * Add DaskTableAccessor (:pr:`648`)
        * Add Schema properties to WoodworkTableAccessor (:pr:`651`)
        * Add KoalasTableAccessor (:pr:`652`)
        * Adds ``__getitem__`` to WoodworkTableAccessor (:pr:`633`)
        * Update Koalas min version and add support for more new pandas dtypes with Koalas (:pr:`678`)
        * Adds ``__setitem__`` to WoodworkTableAccessor (:pr:`669`)
    * Fixes
        * Create new Schema object when performing pandas operation on Accessors (:pr:`595`)
        * Fix bug in ``_reset_semantic_tags`` causing columns to share same semantic tags set (:pr:`666`)
        * Maintain column order in DataFrame and Woodwork repr (:pr:`677`)
    * Changes
        * Move mutual information logic to statistics utils file (:pr:`584`)
        * Bump min Koalas version to 1.4.0 (:pr:`638`)
        * Preserve pandas underlying index when not creating a Woodwork index (:pr:`664`)
        * Restrict Koalas version to ``<1.7.0`` due to breaking changes (:pr:`674`)
        * Clean up dtype usage across Woodwork (:pr:`682`)
        * Improve error when calling accessor properties or methods before init (:pr:`683`)
        * Remove dtype from Schema dictionary (:pr:`685`)
        * Add ``include_index`` param and allow unique columns in Accessor mutual information (:pr:`699`)
        * Include DataFrame equality and ``use_standard_tags`` in WoodworkTableAccessor equality check (:pr:`700`)
        * Remove ``DataTable`` and ``DataColumn`` classes to migrate towards the accessor approach (:pr:`713`)
        * Change ``sample_series`` dtype to not need conversion and remove ``convert_series`` util (:pr:`720`)
        * Rename Accessor methods since ``DataTable`` has been removed (:pr:`723`)
    * Documentation Changes
        * Update README.md and Get Started guide to use accessor (:pr:`655`, :pr:`717`)
        * Update Understanding Types and Tags guide to use accessor (:pr:`657`)
        * Update docstrings and API Reference page (:pr:`660`)
        * Update statistical insights guide to use accessor (:pr:`693`)
        * Update Customizing Type Inference guide to use accessor (:pr:`696`)
        * Update Dask and Koalas guide to use accessor (:pr:`701`)
        * Update index notebook and install guide to use accessor (:pr:`715`)
        * Add section to documentation about schema validity (:pr:`729`)
        * Update README.md and Get Started guide to use ``pd.read_csv`` (:pr:`730`)
        * Make small fixes to documentation formatting (:pr:`731`)
    * Testing Changes
        * Add tests to Accessor/Schema that weren't previously covered (:pr:`712`, :pr:`716`)
        * Update release branch name in notes update check (:pr:`719`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`johnbridstrup`, :user:`tamargrey`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
    * The ``DataTable`` and ``DataColumn`` classes have been removed and replaced by new ``WoodworkTableAccessor`` and ``WoodworkColumnAccessor`` classes which are used through the ``ww`` namespace available on DataFrames after importing Woodwork.

v0.0.11 Mar 15, 2021
====================
    * Changes
        * Restrict Koalas version to ``<1.7.0`` due to breaking changes (:pr:`674`)
        * Include unique columns in mutual information calculations (:pr:`687`)
        * Add parameter to include index column in mutual information calculations (:pr:`692`)
    * Documentation Changes
        * Update to remove warning message from statistical insights guide (:pr:`690`)
    * Testing Changes
        * Update branch reference in tests to run on main (:pr:`641`)
        * Make release notes updated check separate from unit tests (:pr:`642`)
        * Update release branch naming instructions (:pr:`644`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`tamargrey`, :user:`thehomebrewnerd`

v0.0.10 Feb 25, 2021
====================
    * Changes
        * Avoid calculating mutualinfo for non-unique columns (:pr:`563`)
        * Preserve underlying DataFrame index if index column is not specified (:pr:`588`)
        * Add blank issue template for creating issues (:pr:`630`)
    * Testing Changes
        * Update branch reference in tests workflow (:pr:`552`, :pr:`601`)
        * Fixed text on back arrow on install page (:pr:`564`)
        * Refactor test_datatable.py (:pr:`574`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`johnbridstrup`, :user:`tamargrey`

v0.0.9 Feb 5, 2021
==================
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

v0.0.8 Jan 25, 2021
===================
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

v0.0.7 Dec 14, 2020
===================
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

v0.0.6 Nov 30, 2020
===================
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

Breaking Changes
++++++++++++++++
    * The ``DataTable.set_semantic_tags`` method was removed. ``DataTable.set_types`` can be used instead.
    * The ``DataTable.set_logical_types`` method was removed. ``DataTable.set_types`` can be used instead.
    * ``WholeNumber`` was removed from LogicalTypes. Columns that were previously inferred as WholeNumber will now be inferred as Integer.
    * The ``DataTable.get_mutual_information`` was renamed to ``DataTable.mutual_information``.
    * The ``copy_dataframe`` parameter was removed from DataTable initialization.

v0.0.5 Nov 11, 2020
===================
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

Breaking Changes
++++++++++++++++
    * The ``DataColumn.to_pandas`` method was renamed to ``DataColumn.to_series``.
    * The ``DataTable.to_pandas`` method was renamed to ``DataTable.to_dataframe``.
    * ``copy`` is no longer a parameter of ``DataTable.to_dataframe`` or ``DataColumn.to_series``.

v0.0.4 Oct 21, 2020
===================
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

v0.0.3 Oct 9, 2020
==================
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

v0.0.2 Sep 28, 2020
===================
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

v0.1.0 Sep 24, 2020
===================
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
