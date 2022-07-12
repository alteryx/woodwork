from woodwork.statistics_utils.frequency_inference._constants import (
    FREQ_INFERENCE_THRESHOLD,
    NON_INFERABLE_FREQ,
)


def _determine_most_likely_freq(alias_dict, threshold=FREQ_INFERENCE_THRESHOLD):
    """Determine most likely frequency, given the input alias_dict

    Args:
        alias_dict (dict): A dictionary where the key values are a Pandas alias string or a string representing that
            a frequency cannot be inferred. The value of the dictionary is the following:

            - alias (str): The Pandas Freq Alias
            - count (int): The number of windows where this alias is valid
            - min_dt (pd.TimeStamp): The earliest timestamp for this frequency.
            - max_dt (pd.TimeStamp): The latest timestamp for this frequency.

    Returns:
        (list(RangeObject)): A list of RangeObject data objects. A RangeObject has the following properties:

        - dt: an ISO 8601 formatted string of the first NaN timestamp
        - idx: first index of the NaN timestamp. Index is relative to estimated timeseries
        - range: the number of sequential elements that are NaN
    """
    alias_info = alias_dict.values()

    n_total = sum([x["count"] for x in alias_info])
    sorted_aliases = sorted(alias_info, key=lambda item: item["count"], reverse=True)

    most_likely_alias = sorted_aliases[0]["alias"]
    most_likely_count = sorted_aliases[0]["count"]

    if most_likely_count / n_total < threshold:
        return None

    if most_likely_alias is not NON_INFERABLE_FREQ:
        return most_likely_alias
    else:
        return None
