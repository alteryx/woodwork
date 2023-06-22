import contextlib

import numpy as np
import pandas as pd

CONFIG_DEFAULTS = {
    "categorical_threshold": 0.2,
    "numeric_categorical_threshold": None,
    "email_inference_regex": r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)",
    "url_inference_regex": r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)",
    "ipv4_inference_regex": r"(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)",
    "ipv6_inference_regex": "".join(
        r"""
(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]
{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|
([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]
{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}
|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9])
{0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}
[0-9]){0,1}[0-9]))
""".splitlines(),
    ),
    "phone_inference_regex": r"(?:\+?(0{2})?1[-.\s●]?)?\(?([2-9][0-9]{2})\)?[-\.\s●]?([2-9][0-9]{2})[-\.\s●]?([0-9]{4})$",
    "postal_code_inference_regex": r"^[0-9]{5}(?:-[0-9]{4})?$",
    "nan_values": [
        "",
        " ",
        None,
        np.nan,
        pd.NaT,
        "None",
        "NONE",
        "none",
        "NULL",
        "Null",
        "null",
        "NAN",
        "NaN",
        "Nan",
        "nan",
        "NA",
        "na",
        "N/A",
        "n/a",
        "n/A",
        "N/a",
        "<NA>",
        "<N/A>",
        "<n/a>",
        "<na>",
    ],
    "frequence_inference_window_length": 15,
    "frequence_inference_threshold": 0.9,
    "correlation_metrics": ["mutual_info", "pearson", "spearman", "max", "all"],
    "medcouple_threshold": 0.3,  # Must be between 0.0 and 1.0
    "medcouple_sample_size": 10000,
    "boolean_inference_strings": {
        frozenset(["yes", "no"]),
        frozenset(["y", "n"]),
        frozenset(["true", "false"]),
        frozenset(["t", "f"]),
    },
    "boolean_transform_mappings": {
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "true": True,
        "false": False,
        "t": True,
        "f": False,
    },
    # when adding to boolean_inference_ints, add `0, 1` to the set directly
    "boolean_inference_ints": {},
}


class Config:
    def __init__(self, default_values):
        self._defaults = default_values
        self._data = default_values.copy()

    def set_option(self, key, value):
        if key not in self._data.keys():
            raise KeyError(f"Invalid option specified: {key}")
        self._data[key] = value

    def get_option(self, key):
        if key not in self._data.keys():
            raise KeyError(f"Invalid option specified: {key}")
        return self._data[key]

    def reset_option(self, key):
        if key not in self._data.keys():
            raise KeyError(f"Invalid option specified: {key}")
        self._data[key] = self._defaults[key]

    @contextlib.contextmanager
    def with_options(self, **options):
        old_options = {k: self.get_option(k) for k in options}

        for k, v in options.items():
            self.set_option(k, v)
        try:
            yield
        finally:
            for k, v in old_options.items():
                self.set_option(k, v)

    def __repr__(self):
        output_string = "Woodwork Global Config Settings\n"
        output_string += "-" * (len(output_string) - 1)
        for key, value in self._data.items():
            output_string += f"\n{key}: {value}"
        return output_string


config = Config(CONFIG_DEFAULTS)
