import subprocess


def test_list_logical_types():
    subprocess.check_output(['python', '-m', 'woodwork', 'list-logical-types'])


def test_list_semantic_tags():
    subprocess.check_output(['python', '-m', 'woodwork', 'list-semantic-tags'])
