from ._constants import FREQ_INFERENCE_THRESHOLD, NON_INFERABLE_FREQ


def _determine_most_likely_freq(alias_dict):
    n_total = sum(alias_dict.values())
    sorted_freqs = sorted(alias_dict.items(), key=lambda item: item[1], reverse=True)

    most_likely_alias, most_likely_count = sorted_freqs[0]

    if most_likely_count / n_total < FREQ_INFERENCE_THRESHOLD:
        return None

    if most_likely_alias is not NON_INFERABLE_FREQ:
        # Strip off anchor
        most_likely_freq = most_likely_alias.split("-")[0]
        return most_likely_freq
    else:
        return None
