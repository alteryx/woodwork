class ColumnNameMismatchWarning(UserWarning):
    def get_warning_message(self, lose_col, keep_col):
        return f'Name mismatch between {lose_col} and {keep_col}. Series name is now {keep_col}'


class DuplicateTagsWarning(UserWarning):
    def get_warning_message(self, duplicate_tags, name):
        return f"Semantic tag(s) '{', '.join(duplicate_tags)}' already present on column '{name}'"


class StandardTagsRemovalWarning(UserWarning):
    def get_warning_message(self, standard_tags_to_remove, name):
        return f"Removing standard semantic tag(s) '{', '.join(standard_tags_to_remove)}' from column '{name}'"


class UpgradeSchemaWarning(UserWarning):
    def get_warning_message(self, saved_version_str, current_schema_version):
        return ('The schema version of the saved Woodwork table '
                '%s is greater than the latest supported %s. '
                'You may need to upgrade woodwork. Attempting to load Woodwork table ...'
                % (saved_version_str, current_schema_version))


class OutdatedSchemaWarning(UserWarning):
    def get_warning_message(self, saved_version_str):
        return ('The schema version of the saved Woodwork table '
                '%s is no longer supported by this version '
                'of woodwork. Attempting to load Woodwork table ...'
                % (saved_version_str))


class TypingInfoMismatchWarning(UserWarning):
    def get_warning_message(self, attr, invalid_reason, object_type):
        return (f'Operation performed by {attr} has invalidated the Woodwork typing information:\n '
                f'{invalid_reason}.\n '
                f'Please initialize Woodwork with {object_type}.ww.init')


class TypeConversionError(Exception):
    pass


class ParametersIgnoredWarning(UserWarning):
    pass


class ColumnNotPresentError(KeyError):
    pass
