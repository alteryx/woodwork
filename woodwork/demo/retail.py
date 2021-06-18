
import pandas as pd

import woodwork as ww
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage
)


def load_retail(id='demo_retail_data', nrows=None, init_woodwork=True):
    """Load a demo retail dataset into a DataFrame, optionally initializing Woodwork's typing information.

    Args:
        id (str, optional): The name to assign to the DataFrame, if returning a DataFrame with Woodwork
            typing information initialized. If not returning a DataFrame with Woodwork initialized,
            this will be ignored. Defaults to ``demo_retail_data``.
        nrows (int, optional): The number of rows to return in the dataset. If None, will
            return all possible rows. Defaults to None.
        init_woodwork (bool): If True, will return a pandas DataFrame with Woodwork
            typing information initialized. If False, will return a DataFrame without
            Woodwork initialized. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the demo data with Woodwork typing initialized.
        If `init_woodwork` is False, will return an uninitialized DataFrame.
    """
    csv_s3_gz = "https://api.featurelabs.com/datasets/online-retail-logs-2018-08-28.csv.gz?library=woodwork&version=" + ww.__version__
    csv_s3 = "https://api.featurelabs.com/datasets/online-retail-logs-2018-08-28.csv?library=woodwork&version=" + ww.__version__
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
    # Add unique column for index
    df.insert(0, 'order_product_id', range(len(df)))

    if init_woodwork:
        logical_types = {
            'order_product_id': Categorical,
            'order_id': Categorical,
            'product_id': Categorical,
            'description': NaturalLanguage,
            'quantity': Integer,
            'order_date': Datetime,
            'unit_price': Double,
            'customer_name': Categorical,
            'country': Categorical,
            'total': Double,
            'cancelled': Boolean,
        }

        df.ww.init(name=id,
                   index='order_product_id',
                   time_index='order_date',
                   logical_types=logical_types)

    return df
