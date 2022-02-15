from lib2to3.pytree import convert
import pandas as pd

# https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
ALL_ALIASES = [
    {"alias": ["B"], "desc": "business day frequency"},
    {"alias": ["C"], "desc": "custom business day frequency"},
    {"alias": ["D"], "desc": "calendar day frequency"},
    {"alias": ["W"], "desc": "weekly frequency"},
    {"alias": ["M"], "desc": "month end frequency"},
    {"alias": ["SM"], "desc": "semi-month end frequency (15th and end of month)"},
    {"alias": ["BM"], "desc": "business month end frequency"},
    {"alias": ["CBM"], "desc": "custom business month end frequency"},
    {"alias": ["MS"], "desc": "month start frequency"},
    {"alias": ["SMS"], "desc": "semi-month start frequency (1st and 15th)"},
    {"alias": ["BMS"], "desc": "business month start frequency"},
    {"alias": ["CBMS"], "desc": "custom business month start frequency"},
    {"alias": ["Q"], "desc": "quarter end frequency"},
    {"alias": ["BQ"], "desc": "business quarter end frequency"},
    {"alias": ["QS"], "desc": "quarter start frequency"},
    {"alias": ["BQS"], "desc": "business quarter start frequency"},
    {"alias": ["A", "Y"], "desc": "year end frequency"},
    {"alias": ["BA", "BY"], "desc": "business year end frequency"},
    {"alias": ["AS", "YS"], "desc": "year start frequency"},
    {"alias": ["BAS", "BYS"], "desc": "business year start frequency"},
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
    "B": "D",
    "C": "D",
    "SM": None,
    "CBM": "BM",
    "SMS": None,
    "CBMS": "BMS",
    "BH": "H",
}


def case0():
    # 1 hour separation
    # Missing 2005-01-01 20:00:00 at index 20
    dates_1 = pd.date_range("2005-01-01 00:00:00", periods=20, freq="1H")
    dates_2 = pd.date_range("2005-01-01 21:00:00", periods=30, freq="1H")
    dates = dates_1.append(dates_2)

    return {
        "actual_range_start": "2005-01-01T00:00:00",
        "actual_range_end": "2005-01-03T02:00:00",
        "message": None,
        "estimated_freq": "H",
        "estimated_range_start": "2005-01-01T00:00:00",
        "estimated_range_end": "2005-01-03T02:00:00",
        "duplicate_values": [],
        "missing_values": [{"dt": "2005-01-01T20:00:00", "idx": 20, "range": 1}],
        "extra_values": [],
        "data": dates,
    }

    return {
        "name": "1 hour separation",
        "description": "Missing 2005-01-01 20:00:00 at index 20",
        "data": dates,
        "actual_freq": ["H"],
    }


def case1():
    # 2 day separation
    # Missing 2005-02-10 at index 20
    dates_1 = pd.date_range("2005-01-01", periods=20, freq="2D")
    dates_2 = pd.date_range("2005-02-12", periods=30, freq="2D")
    dates = dates_1.append(dates_2)

    return {
        "name": "2 day separation",
        "description": "Missing 2005-02-10 at index 20",
        "data": dates,
        "actual_freq": ["2D"],
    }


def case2():
    # 3 week separation
    # Missing 2006-02-26 at index 20
    dates_1 = pd.date_range("2005-01-02", periods=20, freq="3W")
    dates_2 = pd.date_range("2006-03-19", periods=30, freq="3W")
    dates = dates_1.append(dates_2)

    return {
        "name": "3 week separation",
        "description": " Missing 2006-02-26 at index 20",
        "data": dates,
        "actual_freq": ["3W", "3W-SUN"],
    }


def case3():
    # 1 month start separation
    # Missing 2006-09-01 at index 0
    dates_1 = pd.DatetimeIndex(["2006-08-01"])
    dates_2 = pd.date_range("2006-10-01", periods=30, freq="1MS")
    dates = dates_1.append(dates_2)

    return {
        "name": "1 month start separation",
        "description": "Missing 2006-09-01 at index 0",
        "data": dates,
        "actual_freq": ["MS"],
    }


def case4():
    # 2 month end separation
    # Missing 2006-08-31 at index 0
    dates_1 = pd.DatetimeIndex(["2006-06-30"])
    dates_2 = pd.date_range("2006-10-31", periods=30, freq="2M")
    dates = dates_1.append(dates_2)

    return {
        "name": "2 month end separation",
        "description": "Missing 2006-08-31 at index 0",
        "data": dates,
        "actual_freq": ["2M"],
    }


def case5():
    # 3 year start separation
    # Missing 1943-01-01 at index 0
    dates_1 = pd.DatetimeIndex(["1940-01-01"])
    dates_2 = pd.date_range("1946", periods=30, freq="3AS")
    dates = dates_1.append(dates_2)

    return {
        "name": "3 year start separation",
        "description": "Missing 1943-01-01 at index 0",
        "data": dates,
        "actual_freq": ["3AS"],
    }


def case6():
    # 1 month start separation
    # Missing 2009-04-01 at index 30
    dates_1 = pd.date_range("2006-10-01", periods=30, freq="1MS")
    dates_2 = pd.DatetimeIndex(["2009-05-01"])
    dates = dates_1.append(dates_2)

    return {
        "name": "1 month start separation",
        "description": "Missing 2009-04-01 at index 30",
        "data": dates,
        "actual_freq": ["MS"],
    }


def case7():
    # 2 month end separation
    # Missing 2011-10-31 at index 30
    dates_1 = pd.date_range("2006-10-31", periods=30, freq="2M")
    dates_2 = pd.DatetimeIndex(["2011-12-31"])
    dates = dates_1.append(dates_2)

    return {
        "name": "2 month end separation",
        "description": "Missing 2011-10-31 at index 30",
        "data": dates,
        "actual_freq": ["2M"],
    }


def case8():
    # 3 year end separation
    # Missing 2036-12-31 at index 30
    dates_1 = pd.date_range("1946", periods=30, freq="3A")
    dates_2 = pd.DatetimeIndex(["2039-12-31"])
    dates = dates_1.append(dates_2)

    return {
        "name": "3 year end separation",
        "description": "Missing 2036-12-31 at index 30",
        "data": dates,
        "actual_freq": ["3A", "3A-DEC"],
    }


def case9():
    # 3 hour start separation
    # Missing 2010-04-15 12:00:00 at index 20
    # Missing 2010-04-16 18:00:00 at index 29
    dates_1 = pd.date_range("2010-04-13", periods=20, freq="3H")
    dates_2 = pd.date_range("2010-04-15 15:00:00", periods=9, freq="3H")
    dates_3 = pd.DatetimeIndex(["2010-04-16 21:00:00"])
    dates = dates_1.append(dates_2).append(dates_3)

    return {
        "name": "3 hour start separation",
        "description": "Missing 2010-04-15 12:00:00 at index 20 AND Missing 2010-04-16 18:00:00 at index 29",
        "data": dates,
        "actual_freq": ["3H"],
    }


def case10():
    # 5 day separation
    # Missing 2014-05-25 at index 17
    # Missing 2014-07-14 at index 26
    # Missing 2014-10-02 at index 41
    dates_1 = pd.date_range("2014-03-01", periods=17, freq="5D")
    dates_2 = pd.date_range("2014-05-30", periods=9, freq="5D")
    dates_3 = pd.date_range("2014-07-19", periods=15, freq="5D")
    dates_4 = pd.date_range("2014-10-07", periods=11, freq="5D")
    dates = dates_1.append(dates_2).append(dates_3).append(dates_4)

    return {
        "name": "5 day separation",
        "description": "Missing 2014-05-25 at index 17 AND Missing 2014-07-14 at index 26 AND Missing 2014-10-02 at index 41",
        "data": dates,
        "actual_freq": ["5D"],
    }


def case11():
    # 2 week separation
    # Missing 2010-04-15 12:00:00 at index 20
    # Missing 2010-04-15 15:00:00 at index 21
    dates_1 = pd.date_range("2010-04-13", periods=20, freq="3H")
    dates_2 = pd.date_range("2010-04-15 18:00:00", periods=8, freq="3H")
    dates = dates_1.append(dates_2)

    return {
        "name": "2 week separation",
        "description": "Missing 2010-04-15 12:00:00 at index 20 AND Missing 2010-04-15 15:00:00 at index 21",
        "data": dates,
        "actual_freq": ["3H"],
    }


def case12():
    # 5 day separation
    # Missing 2014-03-06 at index 1
    # Missing 2014-03-11 at index 2
    # Missing 2014-03-16 at index 3
    # Missing 2014-04-30 at index 9
    # Missing 2014-05-05 at index 10
    # Missing 2014-05-10 at index 11
    # Missing 2014-07-19 at index 22
    # Missing 2014-07-24 at index 23
    dates_1 = pd.DatetimeIndex(["2014-03-01"])
    dates_2 = pd.date_range("2014-03-21", periods=8, freq="5D")
    dates_3 = pd.date_range("2014-05-15", periods=13, freq="5D")
    dates_4 = pd.date_range("2014-07-29", periods=11, freq="5D")
    dates = dates_1.append(dates_2).append(dates_3).append(dates_4)

    return {
        "name": "5 day separation",
        "description": "Missing many",
        "data": dates,
        "actual_freq": ["5D"],
    }


def case13():
    # 10 hours separation
    # 2005-01-09 12:00:00 should be 2005-01-09 08:00:00 at index 20
    dates_1 = pd.date_range("2005-01-01 00:00:00", periods=20, freq="10H")
    dates_2 = pd.DatetimeIndex(["2005-01-09 12:00:00"])
    dates_3 = pd.date_range("2005-01-09 18:00:00", periods=29, freq="10H")
    dates = dates_1.append(dates_2).append(dates_3)

    return {
        "name": "10 hours separation",
        "description": "2005-01-09 12:00:00 should be 2005-01-09 08:00:00 at index 20",
        "data": dates,
        "actual_freq": ["10H"],
    }


def case14():
    # 3 months, or quarter start separation
    # 2006-01-12 should be 2006-01-01 at index 20
    dates_1 = pd.date_range("2001-01-01", periods=20, freq="3MS")
    dates_2 = pd.DatetimeIndex(["2006-01-12"])
    dates_3 = pd.date_range("2006-04-01", periods=29, freq="3MS")
    dates = dates_1.append(dates_2).append(dates_3)

    return {
        "name": "3 months, or quarter start separation",
        "description": "2006-01-12 should be 2006-01-01 at index 20",
        "data": dates,
        "actual_freq": ["3MS", "QS", "QS-OCT"],
    }


def case15():
    # 1 semi-month end separation
    # 2001-11-12 should be 2001-11-15 at index 20
    # Note that pandas has trouble actually inferring this as `1SM` frequency, despite accepting
    # if the user manually sets it as dates.freq = '1SM'
    # This might be out of scope as it requires more in-depth checks
    dates_1 = pd.date_range("2001-01-01", periods=20, freq="1SM")
    dates_2 = pd.DatetimeIndex(["2001-11-12"])
    dates_3 = pd.date_range("2001-11-30", periods=29, freq="1SM")
    dates = dates_1.append(dates_2).append(dates_3)

    return {
        "name": "1 semi-month end separation",
        "description": "2001-11-12 should be 2001-11-15 at index 20 (pandas has trouble with this)",
        "data": dates,
        "actual_freq": [None],
    }


def case16():
    # 1 month start separation
    # 2006-09-25 should be 2006-09-01 at index 0
    dates_1 = pd.DatetimeIndex(["2006-09-25"])
    dates_2 = pd.date_range("2006-10-01", periods=30, freq="1MS")
    dates = dates_1.append(dates_2)

    return {
        "name": "1 month start separation",
        "description": "2006-09-25 should be 2006-09-01 at index 0",
        "data": dates,
        "actual_freq": ["MS"],
    }


def case17():
    # 2 month end separation
    # 2006-07-23 should be 2006-08-31 at index 0
    dates_1 = pd.DatetimeIndex(["2006-07-23"])
    dates_2 = pd.date_range("2006-10-31", periods=30, freq="2M")
    dates = dates_1.append(dates_2)

    return {
        "name": "2 month end separation",
        "description": "2006-07-23 should be 2006-08-31 at index 0",
        "data": dates,
        "actual_freq": ["2M"],
    }


def case18():
    # 3 year start separation
    # 1945-07-03 should be 1943-01-01 at index 0
    dates_1 = pd.DatetimeIndex(["1945-07-03"])
    dates_2 = pd.date_range("1946", periods=30, freq="3YS")
    dates = dates_1.append(dates_2)

    return {
        "name": "3 year start separation",
        "description": "1945-07-03 should be 1943-01-01 at index 0",
        "data": dates,
        "actual_freq": ["3AS", "3AS-JAN"],
    }


def case19():
    # 1 month start separation
    # 2009-03-26 should be 2009-04-01 at index 30
    dates_1 = pd.date_range("2006-10-01", periods=30, freq="1MS")
    dates_2 = pd.DatetimeIndex(["2009-03-26"])
    dates = dates_1.append(dates_2)

    return {
        "name": "1 month start separation",
        "description": "2009-03-26 should be 2009-04-01 at index 30",
        "data": dates,
        "actual_freq": ["MS"],
    }


def case20():
    # 2 month end separation
    # 2011-11-12 should be 2011-10-31 at index 30
    dates_1 = pd.date_range("2006-10-31", periods=30, freq="2M")
    dates_2 = pd.DatetimeIndex(["2011-11-12"])
    dates = dates_1.append(dates_2)

    return {
        "name": "2 month end separation",
        "description": "2011-11-12 should be 2011-10-31 at index 30",
        "data": dates,
        "actual_freq": ["2M"],
    }


def case21():
    # 3 year end separation
    # 2034-04-21 should be 2036-12-31 at index 30
    dates_1 = pd.date_range("1946", periods=30, freq="3Y")
    dates_2 = pd.DatetimeIndex(["2034-04-21"])
    dates = dates_1.append(dates_2)

    return {
        "name": "3 year end separation",
        "description": "2034-04-21 should be 2036-12-31 at index 30",
        "data": dates,
        "actual_freq": ["3A", "3A-DEC"],
    }


def case22():
    # 3 hour start separation
    # 2010-04-15 10:00:00 should be 2010-04-15 12:00:00 at index 20
    # 2010-04-16 17:00:00 should be 2010-04-16 15:00:00 at index 29
    dates_1 = pd.date_range("2010-04-13", periods=20, freq="3H")
    dates_2 = pd.DatetimeIndex(["2010-04-15 10:00:00"])
    dates_3 = pd.date_range("2010-04-15 15:00:00", periods=8, freq="3H")
    dates_4 = pd.DatetimeIndex(["2010-04-16 17:00:00"])
    dates = dates_1.append(dates_2).append(dates_3).append(dates_4)

    return {
        "name": "3 hour start separation",
        "description": "2010-04-15 10:00:00 should be 2010-04-15 12:00:00 at index 20 AND 2010-04-16 17:00:00 should be 2010-04-16 15:00:00 at index 29",
        "data": dates,
        "actual_freq": ["3H"],
    }


def case23():
    # 5 day separation
    # 2014-05-23 should be 2014-05-25 at index 17
    # 2014-07-11 should be 2014-07-09 at index 26
    # 2014-09-13 should be 2014-09-17 at index 40
    # 2014-09-18 should be 2014-09-22 at index 41
    dates_1 = pd.date_range("2014-03-01", periods=17, freq="5D")
    dates_2 = pd.DatetimeIndex(["2014-05-23"])
    dates_3 = pd.date_range("2014-05-30", periods=8, freq="5D")
    dates_4 = pd.DatetimeIndex(["2014-07-11"])
    dates_5 = pd.date_range("2014-07-14", periods=13, freq="5D")
    dates_6 = pd.DatetimeIndex(["2014-09-13", "2014-09-18"])
    dates_7 = pd.date_range("2014-09-27", periods=9, freq="5D")
    dates = (
        dates_1.append(dates_2)
        .append(dates_3)
        .append(dates_4)
        .append(dates_5)
        .append(dates_6)
        .append(dates_7)
    )

    return {
        "name": "5 day separation",
        "description": "many",
        "data": dates,
        "actual_freq": ["5D"],
    }


# Duplicate


def case24():
    # 1 month start separation
    # 2006-09-01 is a duplicate at index 0 and should be 2006-08-01
    dates_1 = pd.DatetimeIndex(["2006-09-01"])
    dates_2 = pd.date_range("2006-09-01", periods=30, freq="1MS")
    dates = dates_1.append(dates_2)

    return {
        "name": "1 month start separation",
        "description": "2006-09-01 is a duplicate at index 0 and should be 2006-08-01",
        "data": dates,
        "actual_freq": ["MS"],
    }


def case25():
    # 3 day separation
    # 2001-04-10 is a duplicate at index 30 and should be 2001-04-07
    dates_1 = pd.date_range("2001-01-07", periods=30, freq="3D")
    dates_2 = pd.DatetimeIndex(["2001-04-10", "2001-04-10"])
    dates = dates_1.append(dates_2)

    return {
        "name": "3 day separation",
        "description": "2001-04-10 is a duplicate at index 30 and should be 2001-04-07",
        "data": dates,
        "actual_freq": ["3D"],
    }


def case26():
    # 1 year start separation
    # 2001-01-01 is a duplicate at index 0 and should be 1999-01-01
    # 2001-01-01 is a duplicate at index 1 and should be 2000-01-01
    # 2020-01-01 is a duplicate at index 22 and should be 2021-01-01
    # 2026-01-01 is a duplicate at index 28 and should be 2027-01-01
    dates_1 = pd.DatetimeIndex(["2001-01-01", "2001-01-01"])
    dates_2 = pd.date_range("2001-01-01", periods=20, freq="1AS")
    dates_3 = pd.DatetimeIndex(["2020-01-01"])
    dates_4 = pd.date_range("2022-01-01", periods=5, freq="1AS")
    dates_5 = pd.DatetimeIndex(["2026-01-01"])
    dates = dates_1.append(dates_2).append(dates_3).append(dates_4).append(dates_5)

    return {
        "name": "1 year start separation",
        "description": "many",
        "data": dates,
        "actual_freq": ["AS", "AS-JAN"],
    }


def case27():
    # 3 days separation
    # 2003-01-01 is a duplicate at index 0 and should be 2002-12-29
    # 2003-01-30 should be 2003-01-31 at index 11
    # 2003-02-02 should be 2003-02-03 at index 12
    # Missing 2003-02-21 at index 18
    # Missing 2003-02-24 at index 19
    # 2003-03-16 should be 2003-03-14 at index 23
    dates_1 = pd.DatetimeIndex(["2003-01-01", "2003-01-01"])
    dates_2 = pd.date_range("2003-01-04", periods=9, freq="3D")
    dates_3 = pd.DatetimeIndex(["2003-01-30", "2003-02-02"])
    dates_4 = pd.date_range("2003-02-06", periods=5, freq="3D")
    dates_5 = pd.date_range("2003-02-27", periods=5, freq="3D")
    dates_6 = pd.DatetimeIndex(["2003-03-16", "2003-03-17"])
    dates = (
        dates_1.append(dates_2)
        .append(dates_3)
        .append(dates_4)
        .append(dates_5)
        .append(dates_6)
    )

    return {
        "name": "3 days separation",
        "description": "many",
        "data": dates,
        "actual_freq": ["3D"],
    }


def case28():
    start_dates = pd.date_range("2004-12-25", "2005-01-01", freq="1H")
    dates_1 = pd.DatetimeIndex(["2003-01-01T00:00:00", "2003-01-01"])


def all_pandas_aliases():
    out = []
    for r in ALL_ALIASES:
        alias, desc = r.values()

        converted_alias = [
            KNOWN_FREQ_ISSUES[x] if (x in KNOWN_FREQ_ISSUES) else x for x in alias
        ]
        dates = pd.date_range("2005-01-01 00:00:00", periods=20, freq=alias[0])
        out.append(
            {
                "name": desc,
                "description": f"Checking pandas infer capability on {desc}",
                "data": dates,
                "actual_freq": converted_alias,
            }
        )
    return out


datetime_freq_fixtures = [
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
    case11(),
    case12(),
    case13(),
    case14(),
    case15(),
    case16(),
    case17(),
    case18(),
    case19(),
    case20(),
    case21(),
    case22(),
    case23(),
    case24(),
    case25(),
    case26(),
    case27(),
] + all_pandas_aliases()


def bad_case0():
    # 3 days and 4 days
    # inferred frequency should be None because there isn't enough information
    # to accurately determine the frequency

    dates_1 = pd.date_range("2005-01-01", periods=9, freq="3D")
    dates_2 = pd.date_range("2005-01-01", periods=8, freq="4D")

    dates = dates_1.append(dates_2)

    return {
        "name": "3 days and 4 days",
        "description": "many",
        "data": dates,
        "actual_freq": [None],
    }


bad_dt_freq_fixtures = [bad_case0()]
