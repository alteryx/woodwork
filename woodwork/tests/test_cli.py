import subprocess


def test_list_primitives():
    subprocess.check_output(['python', '-m', 'woodwork', 'list-logical-types'])
