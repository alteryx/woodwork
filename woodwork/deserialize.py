import json
import os


def read_table_metadata(path):
    '''Read datatable metadata from disk, S3 path, or URL.

        Args:
            path (str): Location on disk, S3 path, or URL to read `table_metadata.json`.

        Returns:
            description (dict) : Description of :class:`.Datatable`.
    '''

    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    file = os.path.join(path, 'table_metadata.json')
    with open(file, 'r') as file:
        description = json.load(file)
    description['path'] = path
    return description
