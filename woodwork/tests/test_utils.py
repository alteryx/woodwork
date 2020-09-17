import pytest

from woodwork.utils import _parse_semantic_tags, camel_to_snake


def test_camel_to_snake():
    test_items = {
        'ZIPCode': 'zip_code',
        'SubRegionCode': 'sub_region_code',
        'NaturalLanguage': 'natural_language',
        'Categorical': 'categorical',
    }

    for key, value in test_items.items():
        assert camel_to_snake(key) == value


def test_parse_semantic_tags():
    error_message = "semantic_tags must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _parse_semantic_tags(int, 'semantic_tags')

    error_message = "semantic_tags must be a string, set or list"
    with pytest.raises(TypeError, match=error_message):
        _parse_semantic_tags({'index': {}, 'time_index': {}}, 'semantic_tags')

    error_message = "include parameter must contain only strings"
    with pytest.raises(TypeError, match=error_message):
        _parse_semantic_tags(['index', 1], 'include parameter')

    semantic_tags_from_single = _parse_semantic_tags('index', 'include parameter')
    assert semantic_tags_from_single == {'index'}

    semantic_tags_from_list = _parse_semantic_tags(['index', 'numeric', 'category'], 'semantic_tags')
    assert semantic_tags_from_list == {'index', 'numeric', 'category'}

    semantic_tags_from_set = _parse_semantic_tags({'index', 'numeric', 'category'}, 'include parameter')
    assert semantic_tags_from_set == {'index', 'numeric', 'category'}
