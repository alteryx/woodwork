import os
import tempfile

import woodwork.serialize as serialize


def test_new_file(sample_df_pandas):
    sample_df = sample_df_pandas
    sample_df.ww.init(index='id')
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, 'data'))
        serialize._dump_table(sample_df, tmpdir, index=False, sep=',', encoding='utf-8', engine='python', compression=None)
        file_path = serialize._create_archive(tmpdir)
        os.rename(file_path, "./test_serialization_woodwork_table_schema_{}.tar".format(serialize.SCHEMA_VERSION))
