class ColumnNameMismatchWarning(UserWarning):
    def get_warning_message(self, lose_col, keep_col):
        return f'Name mismatch between {lose_col} and {keep_col}. DataColumn and underlying series name are now {keep_col}'


class DuplicateTagsWarning(UserWarning):
    def get_warning_message(self, duplicate_tags, name):
        return f"Semantic tag(s) '{', '.join(duplicate_tags)}' already present on column '{name}'"


class StandardTagsRemovalWarning(UserWarning):
    def get_warning_message(self, standard_tags_to_remove, name):
        return f"Removing standard semantic tag(s) '{', '.join(standard_tags_to_remove)}' from column '{name}'"


class UpgradeSchemaWarning(UserWarning):
    def get_warning_message(self, saved_version_str, current_schema_version):
        return ('The schema version of the saved woodwork.DataTable '
                '%s is greater than the latest supported %s. '
                'You may need to upgrade woodwork. Attempting to load woodwork.DataTable ...'
                % (saved_version_str, current_schema_version))


class OutdatedSchemaWarning(UserWarning):
    def get_warning_message(self, saved_version_str):
        return ('The schema version of the saved woodwork.DataTable '
                '%s is no longer supported by this version '
                'of woodwork. Attempting to load woodwork.DataTable ...'
                % (saved_version_str))
