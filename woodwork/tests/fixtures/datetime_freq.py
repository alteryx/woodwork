import numpy as np
import pandas as pd

# https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
ALL_ALIASES = [
    {"alias": ["B"], "desc": "business day frequency"},
    {"alias": ["C"], "desc": "custom business day frequency"},
    {"alias": ["D"], "desc": "calendar day frequency"},
    {"alias": ["W-SUN", "W"], "desc": "weekly frequency"},
    {"alias": ["M"], "desc": "month end frequency"},
    {"alias": ["SM"], "desc": "semi-month end frequency (15th and end of month)"},
    {"alias": ["BM"], "desc": "business month end frequency"},
    {"alias": ["CBM"], "desc": "custom business month end frequency"},
    {"alias": ["MS"], "desc": "month start frequency"},
    {"alias": ["SMS"], "desc": "semi-month start frequency (1st and 15th)"},
    {"alias": ["BMS"], "desc": "business month start frequency"},
    {"alias": ["CBMS"], "desc": "custom business month start frequency"},
    {"alias": ["Q-DEC", "Q"], "desc": "quarter end frequency"},
    {"alias": ["BQ-DEC", "BQ"], "desc": "business quarter end frequency"},
    {"alias": ["QS-OCT", "QS"], "desc": "quarter start frequency"},
    {"alias": ["BQS-OCT", "BQS"], "desc": "business quarter start frequency"},
    {"alias": ["A-DEC", "A", "Y"], "desc": "year end frequency"},
    {"alias": ["BA-DEC", "BA", "BY"], "desc": "business year end frequency"},
    {"alias": ["AS-JAN", "AS", "YS"], "desc": "year start frequency"},
    {"alias": ["BAS-JAN", "BAS", "BYS"], "desc": "business year start frequency"},
    {"alias": ["BH"], "desc": "business hour frequency"},
    {"alias": ["H"], "desc": "hourly frequency"},
    {"alias": ["T", "min"], "desc": "minutely frequency"},
    {"alias": ["S"], "desc": "secondly frequency"},
    {"alias": ["L", "ms"], "desc": "milliseconds"},
    {"alias": ["U", "us"], "desc": "microseconds"},
    {"alias": ["N"], "desc": "nanoseconds"},
]

# These keys are inferred to these values by pandas
KNOWN_FREQ_ISSUES = {
    # "B": "D",
    "C": "B",
    "SM": None,
    "CBM": "BM",
    "SMS": None,
    "CBMS": "BMS",
    # "BH": "H",
}

HEAD_RANGE_LEN = 100
TAIL_RANGE_LEN = 100


def pad_datetime_series(dates, freq, pad_start=0, pad_end=100):
    dates = [pd.Timestamp(d) for d in dates]

    head = pd.Series([])
    tail = pd.Series([])

    if pad_start > 0:
        head = (
            pd.date_range(end=dates[0], periods=pad_start, freq=freq)[:-1]
        ).to_series()

    if pad_end > 0:
        tail = (
            pd.date_range(start=dates[-1], periods=pad_end, freq=freq)[1:]
        ).to_series()

    return (
        pd.concat([head, pd.Series(dates), tail])
        .reset_index(drop=True)
        .astype("datetime64[ns]")
    )


def generate_pandas_inferrable():
    dates = ["2005-01-01T00:00:00Z"]

    output = []
    for alias_obj in ALL_ALIASES:
        for freq in alias_obj["alias"]:
            pd_inferred_freq = (
                KNOWN_FREQ_ISSUES[freq]
                if freq in KNOWN_FREQ_ISSUES
                else alias_obj["alias"][0]
            )
            if pd_inferred_freq is not None:
                output.append(
                    {
                        "expected_infer_freq": pd_inferred_freq,
                        "dates": pad_datetime_series(
                            dates, freq=freq, pad_end=TAIL_RANGE_LEN
                        )[1:],
                    }
                )

    return output


def case0():
    """
    missing values
    """

    dates = [
        "00:00:00",
        "01:00:00",
        "02:00:00",
        "04:00:00",  # <-- missing index is here
        "05:00:00",
    ]

    dates = [f"2005-01-01T{d}Z" for d in dates]

    dates = pad_datetime_series(
        dates, freq="H", pad_start=HEAD_RANGE_LEN, pad_end=TAIL_RANGE_LEN
    )

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [],
        "missing_values": [
            {"dt": "2005-01-01T03:00:00", "idx": (HEAD_RANGE_LEN - 1) + 3, "range": 1}
        ],
        "extra_values": [],
        "nan_values": [],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}


def case1():
    """
    duplicate values
    """

    dates = [
        "00:00:00",
        "01:00:00",
        "02:00:00",
        "03:00:00",
        "03:00:00",  # <-- duplicate index starts here
        "03:00:00",
        "04:00:00",
        "05:00:00",
    ]
    dates = [f"2005-01-01T{d}Z" for d in dates]

    dates = pad_datetime_series(
        dates, freq="H", pad_start=HEAD_RANGE_LEN, pad_end=TAIL_RANGE_LEN
    )

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [
            {"dt": "2005-01-01T03:00:00", "idx": (HEAD_RANGE_LEN - 1) + 4, "range": 2}
        ],
        "missing_values": [],
        "extra_values": [],
        "nan_values": [],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}


def case2():
    """
    extra values
    """

    dates = [
        "00:00:00",
        "01:00:00",
        "02:00:00",
        "03:00:00",
        "03:10:00",  # <-- extra index is here
        "04:00:00",
        "05:00:00",
    ]
    dates = [f"2005-01-01T{d}Z" for d in dates]

    dates = pad_datetime_series(
        dates, freq="H", pad_start=HEAD_RANGE_LEN, pad_end=TAIL_RANGE_LEN
    )

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [],
        "missing_values": [],
        "extra_values": [
            {"dt": "2005-01-01T03:10:00", "idx": (HEAD_RANGE_LEN - 1) + 4, "range": 1}
        ],
        "nan_values": [],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}


def case3():
    """
    misaligned values - simple
    """

    dates = [
        "00:00:00",
        "01:00:00",
        "02:00:00",
        "03:10:00",  # <-- missing index and extra index is here
        "04:00:00",
        "05:00:00",
    ]
    dates = [f"2005-01-01T{d}Z" for d in dates]

    dates = pad_datetime_series(
        dates, freq="H", pad_start=HEAD_RANGE_LEN, pad_end=TAIL_RANGE_LEN
    )

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [],
        "missing_values": [
            {"dt": "2005-01-01T03:00:00", "idx": (HEAD_RANGE_LEN - 1) + 3, "range": 1}
        ],
        "extra_values": [
            {"dt": "2005-01-01T03:10:00", "idx": (HEAD_RANGE_LEN - 1) + 3, "range": 1}
        ],
        "nan_values": [],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}


def case4():
    """
    misaligned values - complex
    """

    dates = [
        "00:00:00",
        "01:00:00",
        "01:30:00",
        "02:50:00",
        "03:10:00",
        "04:00:00",
        "05:00:00",
    ]
    dates = [f"2005-01-01T{d}Z" for d in dates]

    dates = pad_datetime_series(
        dates, freq="H", pad_start=HEAD_RANGE_LEN, pad_end=TAIL_RANGE_LEN
    )

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [],
        "missing_values": [
            {"dt": "2005-01-01T02:00:00", "idx": (HEAD_RANGE_LEN - 1) + 2, "range": 2}
        ],
        "extra_values": [
            {"dt": "2005-01-01T01:30:00", "idx": (HEAD_RANGE_LEN - 1) + 2, "range": 3},
        ],
        "nan_values": [],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}


def case5():
    """
    bad start
    """

    head_range = pd.DatetimeIndex(["2004-12-31 23:50:00"])
    tail_range = pd.date_range(start="2005-01-01 01:00:00", periods=100, freq="H")[1:]

    dates = head_range.append(tail_range)

    expected_debug_obj = {
        "actual_range_start": dates[0].isoformat(),
        "actual_range_end": dates[-1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": tail_range[0].isoformat(),
        "estimated_range_end": dates[-1].isoformat(),
        "duplicate_values": [],
        "missing_values": [],
        "extra_values": [
            {"dt": "2004-12-31T23:50:00", "idx": 0, "range": 1},
        ],
        "nan_values": [],
    }

    return {"dates": dates.to_series(), "expected_debug_obj": expected_debug_obj}


def case6():
    """
    nan values
    """

    dates = [
        "2005-01-01T00:00:00.000Z",
        "2005-01-01T01:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T03:00:00.000Z",
        np.nan,
        np.nan,
        "2005-01-01T04:00:00.000Z",
        "2005-01-01T05:00:00.000Z",
    ]

    dates = pad_datetime_series(
        dates, freq="H", pad_start=HEAD_RANGE_LEN, pad_end=TAIL_RANGE_LEN
    )

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [],
        "missing_values": [],
        "extra_values": [],
        "nan_values": [
            {"dt": None, "idx": (HEAD_RANGE_LEN - 1) + 4, "range": 2},
        ],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}


def case7():
    """
    nans and duplicates
    """

    dates = [
        "2005-01-01T00:00:00.000Z",
        "2005-01-01T01:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T03:00:00.000Z",
        np.nan,
        np.nan,
        "2005-01-01T04:00:00.000Z",
        "2005-01-01T05:00:00.000Z",
    ]

    dates = pad_datetime_series(dates, freq="H", pad_start=0, pad_end=TAIL_RANGE_LEN)

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [
            {"dt": "2005-01-01T02:00:00", "idx": 3, "range": 2},
        ],
        "missing_values": [],
        "extra_values": [],
        "nan_values": [
            {"dt": None, "idx": 6, "range": 2},
        ],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}


def case8():
    """
    duplicates and extra
    """

    dates = [
        "2005-01-01T00:00:00.000Z",
        "2005-01-01T01:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T03:00:00.000Z",
        "2005-01-01T03:10:00.000Z",
        "2005-01-01T04:00:00.000Z",
        "2005-01-01T05:00:00.000Z",
    ]

    dates = pad_datetime_series(
        dates, freq="H", pad_start=HEAD_RANGE_LEN, pad_end=TAIL_RANGE_LEN
    )

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [
            {"dt": "2005-01-01T02:00:00", "idx": (HEAD_RANGE_LEN - 1) + 3, "range": 2},
        ],
        "missing_values": [],
        "extra_values": [
            {"dt": "2005-01-01T03:10:00", "idx": (HEAD_RANGE_LEN - 1) + 6, "range": 1},
        ],
        "nan_values": [],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}


def case9():
    """
    duplicates and missing
    """

    dates = [
        "2005-01-01T00:00:00.000Z",
        "2005-01-01T01:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T03:00:00.000Z",
        "2005-01-01T05:00:00.000Z",
    ]

    dates = pad_datetime_series(
        dates, freq="H", pad_start=HEAD_RANGE_LEN, pad_end=TAIL_RANGE_LEN
    )

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [
            {"dt": "2005-01-01T02:00:00", "idx": (HEAD_RANGE_LEN - 1) + 3, "range": 2},
        ],
        "missing_values": [
            {"dt": "2005-01-01T04:00:00", "idx": (HEAD_RANGE_LEN - 1) + 4, "range": 1},
        ],
        "extra_values": [],
        "nan_values": [],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}


def case10(): 
    dates = [
        "2005-01-01T01:00:00.000Z",
        "2005-01-01T01:00:00.000Z",
        "2005-01-01T01:00:00.000Z",
        "2005-01-01T02:00:00.000Z",
        "2005-01-01T03:00:00.000Z",
        "2005-01-01T05:00:00.000Z",
        "2005-01-01T06:00:00.000Z",
        "2005-01-01T06:20:00.000Z",
        "2005-01-01T07:00:00.000Z",
        "2005-01-01T08:00:00.000Z"
    ]

    dates = pad_datetime_series(
        dates, freq="H", pad_start=HEAD_RANGE_LEN, pad_end=TAIL_RANGE_LEN
    )

    expected_debug_obj = {
        "actual_range_start": dates.loc[0].isoformat(),
        "actual_range_end": dates.loc[len(dates) - 1].isoformat(),
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": dates.loc[0].isoformat(),
        "estimated_range_end": dates.loc[len(dates) - 1].isoformat(),
        "duplicate_values": [
            {"dt": "2005-01-01T02:00:00", "idx": (HEAD_RANGE_LEN - 1) + 3, "range": 2},
        ],
        "missing_values": [
            {"dt": "2005-01-01T04:00:00", "idx": (HEAD_RANGE_LEN - 1) + 4, "range": 1},
        ],
        "extra_values": [],
        "nan_values": [],
    }

    return {"dates": dates, "expected_debug_obj": expected_debug_obj}



inferrable_freq_fixtures = [
    case0(),
    case1(),
    case2(),
    case3(),
    case4(),
    case5(),
    case6(),
    case7(),
    case8(),
    case9(),
    case10(),
]
