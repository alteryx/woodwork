import warnings

import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score

from woodwork.data_column import DataColumn
from woodwork.logical_types import (
    Boolean,
    Datetime,
    Double,
    LogicalType,
    str_to_logical_type
)
from woodwork.utils import (
    _convert_input_to_set,
    _get_ltype_class,
    _get_mode,
    _is_numeric_series,
    col_is_datetime
)


class DataTable(object):
    def __init__(self, dataframe,
                 name=None,
                 index=None,
                 time_index=None,
                 semantic_tags=None,
                 logical_types=None,
                 copy_dataframe=False,
                 use_standard_tags=True,
                 make_index=False):
        """Create DataTable

        Args:
            dataframe (pd.DataFrame): Dataframe providing the data for the datatable.
            name (str, optional): Name used to identify the datatable.
            index (str, optional): Name of the index column in the dataframe.
            time_index (str, optional): Name of the time index column in the dataframe.
            semantic_tags (dict, optional): Dictionary mapping column names in the dataframe to the
                semantic tags for the column. The keys in the dictionary should be strings
                that correspond to columns in the underlying dataframe. There are two options for
                specifying the dictionary values:
                (str): If only one semantic tag is being set, a single string can be used as a value.
                (list[str] or set[str]): If multiple tags are being set, a list or set of strings can be
                used as the value.
                Semantic tags will be set to an empty set for any column not included in the
                dictionary.
            logical_types (dict[str -> LogicalType], optional): Dictionary mapping column names in
                the dataframe to the LogicalType for the column. LogicalTypes will be inferred
                for any columns not present in the dictionary.
            copy_dataframe (bool, optional): If True, a copy of the input dataframe will be made
                prior to creating the DataTable. Defaults to False, which results in using a
                reference to the input dataframe.
            use_standard_tags (bool, optional): If True, will add standard semantic tags to columns based
                on the inferred or specified logical type for the column. Defaults to True.
            make_index (bool, optional): If True, will create a new unique, numeric index column with the
                name specified by ``index`` and will add the new index column to the supplied DataFrame.
                If True, the name specified in ``index`` cannot match an existing column name in
                ``dataframe``. If False, the name is specified in ``index`` must match a column
                present in the ``dataframe``. Defaults to False.
        """
        # Check that inputs are valid
        _validate_params(dataframe, name, index, time_index, logical_types, semantic_tags, make_index)

        if copy_dataframe:
            self._dataframe = dataframe.copy()
        else:
            self._dataframe = dataframe

        if make_index:
            self._dataframe.insert(0, index, range(len(self._dataframe)))

        self.name = name
        self.use_standard_tags = use_standard_tags

        # Infer logical types and create columns
        self.columns = self._create_columns(self._dataframe.columns,
                                            logical_types,
                                            semantic_tags,
                                            use_standard_tags)
        if index:
            _update_index(self, index)

        # Update dtypes before setting time index so that any Datetime formatting is applied
        self._update_columns(self.columns)

        if time_index:
            _update_time_index(self, time_index)

    def __getitem__(self, key):
        if isinstance(key, list):
            if not all([isinstance(col, str) for col in key]):
                raise KeyError('Column names must be strings')
            invalid_cols = set(key).difference(set(self.columns.keys()))
            if invalid_cols:
                raise KeyError(f"Column(s) '{', '.join(sorted(list(invalid_cols)))}' not found in DataTable")
            return self._new_dt_from_cols(key)
        if not isinstance(key, str):
            raise KeyError('Column name must be a string')
        if key not in self.columns.keys():
            raise KeyError(f"Column with name '{key}' not found in DataTable")
        return self.columns[key]

    def __setitem__(self, col_name, column):
        if not isinstance(col_name, str):
            raise KeyError('Column name must be a string')

        if not isinstance(column, DataColumn):
            raise ValueError('New column must be of DataColumn type')

        # Don't allow reassigning of index or time index with setitem
        if self.index == col_name:
            raise KeyError('Cannot reassign index. Change column name and then use dt.set_index to reassign index.')
        if self.time_index == col_name:
            raise KeyError('Cannot reassign time index. Change column name and then use dt.set_time_index to reassign time index.')

        if column.name is not None and column.name != col_name:
            warnings.warn(f'Key, {col_name}, does not match the name of the provided DataColumn,'
                          f' {column.name}. Changing DataColumn name to: {col_name}')
            column._series.name = col_name

        self._dataframe[col_name] = column._series
        self._update_columns({col_name: column})

    @property
    def types(self):
        """Dataframe containing the physical dtypes, logical types and semantic
        tags for the table"""
        typing_info = {}
        for dc in self.columns.values():
            typing_info[dc.name] = [dc.dtype, dc.logical_type, dc.semantic_tags]
        df = pd.DataFrame.from_dict(typing_info,
                                    orient='index',
                                    columns=['Physical Type', 'Logical Type', 'Semantic Tag(s)'],
                                    dtype="object")
        df.index.name = 'Data Column'
        return df

    @property
    def ltypes(self):
        """A series listing the logical types for each column in the table"""
        return self.types['Logical Type']

    def _create_columns(self,
                        column_names,
                        logical_types,
                        semantic_tags,
                        use_standard_tags):
        """Create a dictionary with column names as keys and new DataColumn objects
        as values, while assigning any values that are passed for logical types or
        semantic tags to the new column."""
        data_columns = {}
        for name in column_names:
            if logical_types and name in logical_types:
                logical_type = logical_types[name]
            else:
                logical_type = None
            if semantic_tags and name in semantic_tags:
                semantic_tag = semantic_tags[name]
            else:
                semantic_tag = None
            dc = DataColumn(self._dataframe[name], logical_type, semantic_tag, use_standard_tags)
            data_columns[dc.name] = dc
        return data_columns

    @property
    def logical_types(self):
        """A dictionary containing logical types for each column"""
        return {dc.name: dc.logical_type for dc in self.columns.values()}

    @property
    def physical_types(self):
        """A dictionary containing physical types for each column"""
        return {dc.name: dc.dtype for dc in self.columns.values()}

    @property
    def semantic_tags(self):
        """A dictionary containing semantic tags for each column"""
        return {dc.name: dc.semantic_tags for dc in self.columns.values()}

    @property
    def index(self):
        """The index column for the table"""
        for column in self.columns.values():
            if 'index' in column.semantic_tags:
                return column.name
        return None

    @index.setter
    def index(self, index):
        if self.index and index is None:
            updated_index_col = self.columns[self.index].remove_semantic_tags('index')
            self._update_columns({self.index: updated_index_col})
        elif index is not None:
            _update_index(self, index, self.index)

    @property
    def time_index(self):
        """The time index column for the table"""
        for column in self.columns.values():
            if 'time_index' in column.semantic_tags:
                return column.name
        return None

    @time_index.setter
    def time_index(self, time_index):
        if self.time_index and time_index is None:
            updated_time_index_col = self.columns[self.time_index].remove_semantic_tags('time_index')
            self._update_columns({self.time_index: updated_time_index_col})
        elif time_index is not None:
            _update_time_index(self, time_index, self.time_index)

    def _update_columns(self, new_columns):
        """Update the DataTable columns based on items contained in the
        provided new_columns dictionary"""
        for name, column in new_columns.items():
            self.columns[name] = column
            # Make sure the underlying dataframe is in sync in case series data has changed
            self._dataframe[name] = column._series

    def pop(self, column_name):
        """Return a DataColumn and drop it from the DataTable.

        Args:
            column (str): Name of the column to pop.

        Returns:
            woodwork.DataColumn: DataColumn including logical type and semantic tags.
        """
        col = self[column_name]
        del self.columns[column_name]
        self._dataframe.drop(column_name, axis=1, inplace=True)
        return col

    def set_index(self, index):
        """Set the index column and return a new DataTable. Adds the 'index' semantic
        tag to the column and clears the tag from any previously set index column.
        Setting a column as the index column will also cause any previously set standard
        tags for the column to be removed.

        Args:
            index (str): The name of the column to set as the index

        Returns:
            woodwork.DataTable: DataTable with the specified index column set.
        """
        new_dt = self._new_dt_from_cols(self.columns)
        _update_index(new_dt, index, self.index)
        return new_dt

    def set_time_index(self, time_index):
        """Set the time index column. Adds the 'time_index' semantic tag to the column and
        clears the tag from any previously set index column

        Args:
            time_index (str): The name of the column to set as the time index.
        """
        new_dt = self._new_dt_from_cols(self.columns)
        _update_time_index(new_dt, time_index, self.time_index)
        return new_dt

    def set_logical_types(self, logical_types, retain_index_tags=True):
        """Update the logical type for any columns names in the provided logical_types
        dictionary. Replaces existing columns with new DataColumn objects and returns a new
        DataTable object.

        Args:
            logical_types (dict[str -> str/list/set]): A dictionary defining the new logical types for the
                specified columns.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If false, will clear all semantic tags. Defaults to
                True.

        Returns:
            woodwork.DataTable: DataTable with updated logical types
        """
        _check_logical_types(self._dataframe, logical_types)
        new_dt = self._update_cols_and_get_new_dt('set_logical_type', logical_types, retain_index_tags)
        return new_dt

    def add_semantic_tags(self, semantic_tags):
        """Adds specified semantic tags to columns. Will retain any previously set values.
        Replaces updated columns with new DataColumn objects and returns a new DataTable object.

        Args:
            semantic_tags (dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataTable to the tags that should be added to the column

        Returns:
            woodwork.DataTable: DataTable with semantic tags added
        """
        _check_semantic_tags(self._dataframe, semantic_tags)
        return self._update_cols_and_get_new_dt('add_semantic_tags', semantic_tags)

    def remove_semantic_tags(self, semantic_tags):
        """Remove the semantic tags for any column names in the provided semantic_tags
        dictionary. Replaces the column with a new DataColumn object and return a new DataTable
        object.

        Args:
            semantic_tags (dict[str -> str/list/set]): A dictionary mapping the columns
                in the DataTable to the tags that should be removed to the column

        Returns:
            woodwork.DataTable: DataTable with the specified semantic tags removed
        """
        _check_semantic_tags(self._dataframe, semantic_tags)
        return self._update_cols_and_get_new_dt('remove_semantic_tags', semantic_tags)

    def set_semantic_tags(self, semantic_tags, retain_index_tags=True):
        """Update the semantic tags for any column names in the provided semantic_tags
        dictionary. Replaces the existing semantic tags with the new values. Also replaces
        any updated columns with new DataColumn objects and returns a new DataTable object.

        Args:
            semantic_tags (dict): A dictionary defining the new semantic_tags for the
                specified columns.
            retain_index_tags (bool, optional): If True, will retain any index or
                time_index semantic tags set on the column. If False, will replace all
                semantic tags. Defaults to True.

        Returns:
            woodwork.DataTable: DataTable with the specified semantic tags set
        """
        _check_semantic_tags(self._dataframe, semantic_tags)
        return self._update_cols_and_get_new_dt('set_semantic_tags', semantic_tags, retain_index_tags)

    def reset_semantic_tags(self, columns=None, retain_index_tags=False):
        """Reset the semantic tags for the specified columns to the default values and
        return a new DataTable. The default values will be either an empty set or a set
        of the standard tags based on the column logical type, controlled by the
        use_standard_tags property on the table. Columns names can be provided as a
        single string, a list of strings or a set of strings. If columns is not specified,
        tags will be reset for all columns.

        Args:
            columns (str/list/set): The columns for which the semantic tags should be reset.
            retain_index_tags (bool, optional): If True, will retain any index or time_index
                semantic tags set on the column. If False, will clear all semantic tags. Defaults to
                False.

        Returns:
            woodwork.DataTable: DataTable with semantic tags reset to default values
        """
        columns = _convert_input_to_set(columns, "columns")
        cols_not_found = sorted(list(columns.difference(set(self._dataframe.columns))))
        if cols_not_found:
            raise LookupError("Input contains columns that are not present in "
                              f"dataframe: '{', '.join(cols_not_found)}'")
        if not columns:
            columns = self.columns.keys()
        return self._update_cols_and_get_new_dt('reset_semantic_tags', columns, retain_index_tags)

    def _update_cols_and_get_new_dt(self, method, new_values, *args):
        """Helper method that can be used for updating columns by calling the column method
        that is specified, passing along information contained in new_values and any
        additional positional arguments.

        Args:
            method (str): The name of the method to call on the DataColumn object to perform the update.
            new_values (dict/list): If a dictionary is provided the keys should correspond to column
                names and the items in the dictionary values will be passed along to the DataColumn method.
                If a list is provided, the items in the list should correspond to column names and
                no additional values will be passed along to the DataColumn method.

        Returns:
            woodwork.DataTable: A new DataTable with updated columns
        """
        new_dt = self._new_dt_from_cols(self.columns)
        cols_to_update = {}
        if isinstance(new_values, dict):
            for name, tags in new_values.items():
                cols_to_update[name] = getattr(new_dt.columns[name], method)(tags, *args)
        else:
            for name in new_values:
                cols_to_update[name] = getattr(new_dt.columns[name], method)(*args)
        new_dt._update_columns(cols_to_update)
        return new_dt

    def to_pandas(self, copy=False):
        """Retrieves the DataTable's underlying dataframe.

        Note: Do not modify the dataframe unless copy=True has been set to avoid unexpected behavior

        Args:
            copy (bool): If set to True, returns a copy of the underlying dataframe.
                If False, will return a reference to the DataTable's dataframe, which,
                if modified, can cause unexpected behavior in the DataTable.
                Defaults to False.

        Returns:
            pandas.DataFrame: The underlying dataframe of the DataTable
        """
        if copy:
            return self._dataframe.copy()
        return self._dataframe

    def select(self, include):
        """Create a DataTable including only columns whose logical type and
        semantic tags are specified in the list of types and tags to include.

        Args:
            include (str or LogicalType or list[str or LogicalType]): Logical
                types, semantic tags or column names to include
                in the DataTable.

        Returns:
            DataTable: The subset of the original DataTable that contains just the
            logical types and semantic tags in ``include``.
        """
        cols_to_include = self._filter_cols(include)
        return self._new_dt_from_cols(cols_to_include)

    def _filter_cols(self, include, col_names=False):
        """Return list of columns filtered in specified way. In case of collision, favors logical types
        then semantic tag then column name.

        Args:
            include (str or LogicalType or list[str or LogicalType]): parameter or list of parameters to
                filter columns by.

            col_names (bool): Specifies whether to filter columns by name. Defaults to False.

        Returns:
            List[str] of column names that fit into filter.
        """
        if not isinstance(include, list):
            include = [include]

        ltypes_used = set()
        ltypes_in_dt = {_get_ltype_class(col.logical_type) for col in self.columns.values()}

        tags_used = set()
        tags_in_dt = {tag for col in self.columns.values() for tag in col.semantic_tags}

        unused_selectors = []

        cols_to_include = set()

        for selector in include:
            if _get_ltype_class(selector) in LogicalType.__subclasses__():
                if selector not in LogicalType.__subclasses__():
                    raise TypeError(f"Invalid selector used in include: {selector} cannot be instantiated")
                if selector in ltypes_in_dt:
                    ltypes_used.add(selector)
                else:
                    unused_selectors.append(str(selector))
            elif isinstance(selector, str):
                # If the str is a viable ltype, it'll take precedence
                # but if it's not present, we'll check if it's a tag
                ltype = str_to_logical_type(selector, raise_error=False)
                if ltype and ltype in ltypes_in_dt:
                    ltypes_used.add(ltype)
                    continue
                elif selector in tags_in_dt:
                    tags_used.add(selector)
                elif col_names and selector in self.columns:
                    cols_to_include.add(selector)
                else:
                    unused_selectors.append(selector)
            else:
                raise TypeError(f"Invalid selector used in include: {selector} must be either a string or LogicalType")

        if unused_selectors:
            not_present_str = ', '.join(sorted(unused_selectors))
            warnings.warn(f'The following selectors were not present in your DataTable: {not_present_str}')

        for col_name, col in self.columns.items():
            if _get_ltype_class(col.logical_type) in ltypes_used or col.semantic_tags.intersection(tags_used):
                cols_to_include.add(col_name)

        return list(cols_to_include)

    def _new_dt_from_cols(self, cols_to_include):
        """Creates a new DataTable from a list of column names, retaining all types,
        indices, and name of original DataTable"""
        assert all([col_name in self.columns for col_name in cols_to_include])

        new_semantic_tags = {col_name: semantic_tag_set for col_name, semantic_tag_set
                             in self.semantic_tags.items() if col_name in cols_to_include}
        new_logical_types = {col_name: logical_type for col_name, logical_type
                             in self.logical_types.items() if col_name in cols_to_include}
        new_index = self.index if self.index in cols_to_include else None
        new_time_index = self.time_index if self.time_index in cols_to_include else None
        # Remove 'index' or 'time_index' from semantic tags, if present as those can't be set directly during init
        if new_index:
            new_semantic_tags[new_index] = new_semantic_tags[new_index].difference({'index'})
        if new_time_index:
            new_semantic_tags[new_time_index] = new_semantic_tags[new_time_index].difference({'time_index'})

        return DataTable(self._dataframe.loc[:, cols_to_include],
                         name=self.name,
                         index=new_index,
                         time_index=new_time_index,
                         semantic_tags=new_semantic_tags,
                         logical_types=new_logical_types,
                         copy_dataframe=False,
                         use_standard_tags=self.use_standard_tags)

    def describe(self, include=None):
        """Calculates statistics for data contained in DataTable.
        Arguments:
            include (list[str or LogicalType], optional): filter for what columns to include in the
            statistics returned. Can be a list of columns, semantic tags, logical types, or a list
            combining any of the three. It follows the most broad specification. Favors logical types
            then semantic tag then column name.

        Returns:
            pd.DataFrame: A Dataframe containing statistics for the data or the subset of the original
            DataTable that just containing the logical types, semantic tags, or column names specified
            in ``include``.
        """
        agg_stats_to_calculate = {
            'category': ["count", "nunique"],
            'numeric': ["count", "max", "min", "nunique", "mean", "std"],
            Datetime: ["count", "max", "min", "nunique", "mean"],
        }
        if include is not None:
            filtered_cols = self._filter_cols(include, col_names=True)
            if filtered_cols == []:
                raise ValueError('no columns matched the given include filters.')
            cols_to_include = [(k, v) for k, v in self.columns.items() if k in filtered_cols]
        else:
            cols_to_include = self.columns.items()

        results = {}

        for column_name, column in cols_to_include:
            if 'index' in column.semantic_tags:
                continue
            values = {}
            logical_type = column.logical_type
            semantic_tags = column.semantic_tags
            series = column._series

            # Calculate Aggregation Stats
            if column._is_categorical():
                agg_stats = agg_stats_to_calculate['category']
            elif column._is_numeric():
                agg_stats = agg_stats_to_calculate['numeric']
            elif _get_ltype_class(logical_type) == Datetime:
                agg_stats = agg_stats_to_calculate[Datetime]
            else:
                agg_stats = ["count"]
            values = series.agg(agg_stats).to_dict()

            # Calculate other specific stats based on logical type or semantic tags
            if _get_ltype_class(logical_type) == Boolean:
                values["num_false"] = series.value_counts().get(False, 0)
                values["num_true"] = series.value_counts().get(True, 0)
            elif column._is_numeric():
                quant_values = series.quantile([0.25, 0.5, 0.75]).tolist()
                values["first_quartile"] = quant_values[0]
                values["second_quartile"] = quant_values[1]
                values["third_quartile"] = quant_values[2]

            values["nan_count"] = series.isna().sum()
            values["mode"] = _get_mode(series)
            values["physical_type"] = column.dtype
            values["logical_type"] = logical_type
            values["semantic_tags"] = semantic_tags
            results[column_name] = values

        index_order = [
            'physical_type',
            'logical_type',
            'semantic_tags',
            'count',
            'nunique',
            'nan_count',
            'mean',
            'mode',
            'std',
            'min',
            'first_quartile',
            'second_quartile',
            'third_quartile',
            'max',
            'num_true',
            'num_false',
        ]
        return pd.DataFrame(results).reindex(index_order)

    def _handle_nans_for_mutual_info(self, data):
        """
        Remove NaN values in the dataframe so that mutual information can be calculated

        Args:
            data (pd.DataFrame): dataframe to use for caculating mutual information

        Returns:
            pd.DataFrame: data with fully null columns removed and nans filled in
                with either mean or mode

        """
        # remove fully null columns
        data = data.loc[:, data.columns[data.notnull().any()]]

        # replace or remove null values
        for column_name in data.columns[data.isnull().any()]:
            column = self[column_name]
            series = column._series
            ltype = column._logical_type

            if column._is_numeric():
                mean = series.mean()
                if isinstance(mean, float) and not _get_ltype_class(ltype) == Double:
                    data[column_name] = series.astype('float')
                data[column_name] = series.fillna(mean)
            elif column._is_categorical() or _get_ltype_class(ltype) == Boolean:
                mode = _get_mode(series)
                data[column_name] = series.fillna(mode)
        return data

    def _make_categorical_for_mutual_info(self, data, num_bins):
        """Transforms dataframe columns into numeric categories so that
        mutual information can be calculated

        Args:
            data (pd.DataFrame): dataframe to use for caculating mutual information
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical.


        Returns:
            data (pd.DataFrame): Transformed data
        """

        col_names = data.columns.to_list()
        for col_name in col_names:
            if self[col_name]._is_numeric():
                # bin numeric features to make categories
                data[col_name] = pd.qcut(data[col_name], num_bins, duplicates="drop")
            # convert categories to integers
            new_col = data[col_name]
            if new_col.dtype != 'category':
                new_col = new_col.astype('category')
            data[col_name] = new_col.cat.codes
        return data

    def get_mutual_information(self, num_bins=10, nrows=None):
        """
        Calculates mutual information between all pairs of columns in the DataTable
        that support mutual information. Logical Types that support mutual information are
        as follows:  Boolean, Categorical, CountryCode, Double, Integer, Ordinal, SubRegionCode,
        WholeNumber, and ZIPCode

        Args:
            num_bins (int): Determines number of bins to use for converting
                numeric features into categorical.
            nrows (int): The number of rows to sample for when determining mutual info.
                If specified, samples the desired number of rows from the data.
                Defaults to using all rows.

        Returns:
            pd.DataFrame: A Dataframe containing mutual information with columns `column_1`,
            `column_2`, and `mutual_info` that is sorted in decending order by mutual info.
            Mutual information values are between 0 (no mutual information) and 1
            (perfect correlation).
        """
        # We only want Numeric, Categorical, and Boolean columns
        # And we don't want the index column
        valid_columns = {col_name for col_name, column
                         in self.columns.items() if (col_name != self.index and
                                                     (column._is_numeric() or
                                                      column._is_categorical() or
                                                      _get_ltype_class(column.logical_type) == Boolean)
                                                     )}
        data = self._dataframe[valid_columns]

        # cut off data if necessary
        if nrows is not None and nrows < data.shape[0]:
            data = data.sample(nrows)

        data = self._handle_nans_for_mutual_info(data)
        data = self._make_categorical_for_mutual_info(data, num_bins)

        # calculate mutual info for all pairs of columns
        mutual_info = []
        col_names = data.columns.to_list()
        for i, a_col in enumerate(col_names):
            for j in range(i, len(col_names)):
                b_col = col_names[j]
                if a_col == b_col:
                    # Ignore because the mutual info for a column with itself will always be 1
                    continue
                else:
                    mi_score = normalized_mutual_info_score(data[a_col], data[b_col])
                    mutual_info.append(
                        {"column_1": a_col, "column_2": b_col, "mutual_info": mi_score}
                    )
        mi = pd.DataFrame(mutual_info)
        if not mutual_info:
            return mi
        return mi.sort_values('mutual_info', ascending=False)


def _validate_params(dataframe, name, index, time_index, logical_types, semantic_tags, make_index):
    """Check that values supplied during DataTable initialization are valid"""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError('Dataframe must be a pandas.DataFrame')
    _check_unique_column_names(dataframe)
    if name and not isinstance(name, str):
        raise TypeError('DataTable name must be a string')
    if index or make_index:
        _check_index(dataframe, index, make_index)
    if logical_types:
        _check_logical_types(dataframe, logical_types)
    if time_index:
        datetime_format = None
        logical_type = None
        if logical_types is not None and time_index in logical_types:
            logical_type = logical_types[time_index]
            if _get_ltype_class(logical_types[time_index]) == Datetime:
                datetime_format = logical_types[time_index].datetime_format

        _check_time_index(dataframe, time_index, datetime_format=datetime_format, logical_type=logical_type)

    if semantic_tags:
        _check_semantic_tags(dataframe, semantic_tags)


def _check_unique_column_names(dataframe):
    if not dataframe.columns.is_unique:
        raise IndexError('Dataframe cannot contain duplicate columns names')


def _check_index(dataframe, index, make_index=False):
    if index and not isinstance(index, str):
        raise TypeError('Index column name must be a string')
    if not make_index and index not in dataframe.columns:
        # User specifies an index that is not in the dataframe, without setting make_index to True
        raise LookupError(f'Specified index column `{index}` not found in dataframe. To create a new index column, set make_index to True.')
    if index and not make_index and not dataframe[index].is_unique:
        # User specifies an index that is in the dataframe but not unique
        raise IndexError('Index column must be unique')
    if make_index and index and index in dataframe.columns:
        # User sets make_index to True, but supplies an index name that matches a column already present
        raise IndexError('When setting make_index to True, the name specified for index cannot match an existing column name')
    if make_index and not index:
        # User sets make_index to True, but does not supply a name for the index
        raise IndexError('When setting make_index to True, the name for the new index must be specified in the index parameter')


def _check_time_index(dataframe, time_index, datetime_format=None, logical_type=None):
    if not isinstance(time_index, str):
        raise TypeError('Time index column name must be a string')
    if time_index not in dataframe.columns:
        raise LookupError(f'Specified time index column `{time_index}` not found in dataframe')
    if not (_is_numeric_series(dataframe[time_index], logical_type) or
            col_is_datetime(dataframe[time_index], datetime_format=datetime_format)):
        raise TypeError('Time index column must contain datetime or numeric values')


def _check_logical_types(dataframe, logical_types):
    if not isinstance(logical_types, dict):
        raise TypeError('logical_types must be a dictionary')
    cols_not_found = set(logical_types.keys()).difference(set(dataframe.columns))
    if cols_not_found:
        raise LookupError('logical_types contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')


def _check_semantic_tags(dataframe, semantic_tags):
    if not isinstance(semantic_tags, dict):
        raise TypeError('semantic_tags must be a dictionary')
    cols_not_found = set(semantic_tags.keys()).difference(set(dataframe.columns))
    if cols_not_found:
        raise LookupError('semantic_tags contains columns that are not present in '
                          f'dataframe: {sorted(list(cols_not_found))}')


def _update_index(data_table, index, old_index=None):
    """Add the `index` tag to the specified index column and remove the tag from the
    old_index column, if specified. Also checks that the specified index column
    can be used as an index."""
    _check_index(data_table._dataframe, index)
    data_table.columns[index]._set_as_index()
    if old_index:
        data_table._update_columns({old_index: data_table.columns[old_index].remove_semantic_tags('index')})


def _update_time_index(data_table, time_index, old_time_index=None):
    """Add the `time_index` tag to the specified time_index column and remove the tag from the
    old_time_index column, if specified. Also checks that the specified time_index
    column can be used as a time index."""
    _check_time_index(data_table._dataframe, time_index)
    data_table.columns[time_index]._set_as_time_index()
    if old_time_index:
        data_table._update_columns({old_time_index: data_table.columns[old_time_index].remove_semantic_tags('time_index')})
