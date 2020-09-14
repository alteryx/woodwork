import pandas as pd

import data_tables as dt


def load_retail(id='demo_retail_data', nrows=None, return_single_table=False):
    csv_s3_gz = "https://api.featurelabs.com/datasets/online-retail-logs-2018-08-28.csv.gz?version=" + dt.__version__
    csv_s3 = "https://api.featurelabs.com/datasets/online-retail-logs-2018-08-28.csv?version=" + dt.__version__
    # Try to read in gz compressed file
    try:
        df = pd.read_csv(csv_s3_gz,
                         nrows=nrows,
                         parse_dates=["order_date"])
    # Fall back to uncompressed
    except Exception:
        df = pd.read_csv(csv_s3,
                         nrows=nrows,
                         parse_dates=["order_date"])
    return df
