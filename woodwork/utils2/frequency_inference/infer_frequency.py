from .generate_freq_candidates import _generate_freq_candidates
from .determine_most_likely_freq import _determine_most_likely_freq
from .build_freq_dataframe import _build_freq_dataframe
from .generate_estimated_timeseries import _generate_estimated_timeseries
from .determine_missing_values import _determine_missing_values
from .determine_duplicate_values import _determine_duplicate_values
from .determine_extra_values import _determine_extra_values


def infer_frequency(observed_ts):
    # Generate Frequency Candidates

    observed_ts_no_dupes = observed_ts.drop_duplicates()
    candidate_df, alias_dict = _generate_freq_candidates(observed_ts_no_dupes)

    most_likely_freq = _determine_most_likely_freq(alias_dict)

    if most_likely_freq is None:
        print("Freq cannot be inferred")
        return

    # Build Freq Dataframe, get alias_dict
    freq_df = _build_freq_dataframe(candidate_df)

    estimated_ts = _generate_estimated_timeseries(freq_df, most_likely_freq)

    missing = _determine_missing_values(estimated_ts, observed_ts)
    print("Missing")
    print(missing)

    extra = _determine_extra_values(estimated_ts, observed_ts)
    print("Extra")
    print(extra)

    duplicates = _determine_duplicate_values(observed_ts)
    print("Duplicate")
    print(duplicates)
