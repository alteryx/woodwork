import re


def camel_to_snake(s):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def _parse_semantic_tags(semantic_tags, error_language):
    if not semantic_tags:
        return set()

    if type(semantic_tags) not in [list, set, str]:
        raise TypeError(f"{error_language} must be a string, set or list")

    if isinstance(semantic_tags, str):
        return {semantic_tags}

    if isinstance(semantic_tags, list):
        semantic_tags = set(semantic_tags)

    if not all([isinstance(tag, str) for tag in semantic_tags]):
        raise TypeError(f"{error_language} must contain only strings")

    return semantic_tags
