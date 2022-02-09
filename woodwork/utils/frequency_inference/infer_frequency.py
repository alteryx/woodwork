from generate_freq_candidates import generate_freq_candidates
from determine_most_likely_freq import determine_most_likely_freq
from build_freq_dataframe import build_freq_dataframe
from generate_estimated_timeseries import generate_estimated_timeseries


def infer_frequency(observed_ts):
    # Generate Frequency Candidates

    observed_ts_no_dupes = observed_ts.drop_duplicates()
    candidate_df, alias_dict = generate_freq_candidates(observed_ts_no_dupes)

    most_likely_freq = determine_most_likely_freq(alias_dict)

    if most_likely_freq is None:
        print("Freq cannot be inferred")
        return

    # Build Freq Dataframe, get alias_dict
    freq_df = build_freq_dataframe(candidate_df)

    estimated_ts = generate_estimated_timeseries(freq_df, most_likely_freq)

    missing = determine_missing(estimated_ts, observed_ts)
    print("Missing")
    print(missing)

    extra = determine_extra(estimated_ts, observed_ts)
    print("Extra")
    print(extra)

    duplicates = determine_duplicates(observed_ts)
    print("Duplicate")
    print(duplicates)
