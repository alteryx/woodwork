from ._constants import FREQ_INFERENCE_THRESHOLD, NON_INFERABLE_FREQ


def _determine_most_likely_freq(alias_dict, threshold=FREQ_INFERENCE_THRESHOLD):

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
