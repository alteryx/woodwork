class DuplicateTagsWarning(UserWarning):
    def get_warning_message(self, duplicate_tags, name):
        return f"Semantic tag(s) '{', '.join(duplicate_tags)}' already present on column '{name}'"


class StandardTagsChangedWarning(UserWarning):
    def get_warning_message(self, use_standard_tags, col_name=None):
        changed = "added to" if use_standard_tags else "removed from"
        name = ('"' + col_name + '"') if col_name is not None else "your column"
        return f"Standard tags have been {changed} {name}"


class UpgradeSchemaWarning(UserWarning):
    def get_warning_message(self, saved_version_str, current_schema_version):
        return (
            "The schema version of the saved Woodwork table "
            "%s is greater than the latest supported %s. "
            "You may need to upgrade woodwork. Attempting to load Woodwork table ..."
            % (saved_version_str, current_schema_version)
        )


class OutdatedSchemaWarning(UserWarning):
    def get_warning_message(self, saved_version_str):
        return (
            "The schema version of the saved Woodwork table "
            "%s is no longer supported by this version "
            "of woodwork. Attempting to load Woodwork table ..." % (saved_version_str)
        )


class IndexTagRemovedWarning(UserWarning):
    pass


class TypingInfoMismatchWarning(UserWarning):
    def get_warning_message(self, attr, invalid_reason, object_type):
        return (
            f"Operation performed by {attr} has invalidated the Woodwork typing information:\n "
            f"{invalid_reason}.\n "
            f"Please initialize Woodwork with {object_type}.ww.init"
        )


class TypeConversionError(Exception):
    def __init__(self, series, new_dtype, logical_type):
        message = f"Error converting datatype for {series.name} from type {str(series.dtype)} "
        message += f"to type {new_dtype}. Please confirm the underlying data is consistent with "
        message += f"logical type {logical_type}."
        super().__init__(message)


class TypeConversionWarning(UserWarning):
    pass


class ParametersIgnoredWarning(UserWarning):
    pass


class ColumnNotPresentError(KeyError):
    def __init__(self, column):
        if isinstance(column, str):
            return super().__init__(
                f"Column with name '{column}' not found in DataFrame",
            )
        elif isinstance(column, list):
            return super().__init__(f"Column(s) '{column}' not found in DataFrame")


class ColumnNotPresentInSchemaError(KeyError):
    def __init__(self, column):
        return super().__init__(
            f"Column(s) {column} not found in Woodwork schema. Please initialize with DataFrame.ww.init",
        )


class ColumnBothIgnoredAndSetError(KeyError):
    pass


class WoodworkNotInitError(AttributeError):
    pass


class WoodworkNotInitWarning(UserWarning):
    pass


class TypeValidationError(Exception):
    pass


class SparseDataWarning(UserWarning):
    pass


class WoodworkFileExistsError(FileExistsError):
    pass
