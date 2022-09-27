import warnings

from woodwork.config import CONFIG_DEFAULTS
from woodwork.exceptions import ParametersIgnoredWarning


def _parse_measures(measures):
    """Validates the provided measures argument and returns the list of measures
    included in the final dataframe/dictionary, an ordered list of measures to
    calculate, and whether to calculate the max score across measures."""
    if not isinstance(measures, list):
        if not isinstance(measures, str):
            raise TypeError(f"Supplied measure {measures} is not a string")
        measures = [measures]

    if len(measures) == 0:
        raise ValueError("No measures supplied")

    calc_pearson = False
    calc_mutual = False
    calc_spearman = False
    calc_max = False
    calc_order = []
    for measure in measures:
        if measure == "all":
            if not measures == ["all"]:
                warnings.warn(
                    ParametersIgnoredWarning(
                        "additional measures to 'all' measure found; 'all' should be used alone",
                    ),
                )
            measures = ["max", "pearson", "spearman", "mutual_info"]
            calc_pearson = True
            calc_mutual = True
            calc_spearman = True
            calc_max = True
            break
        elif measure == "pearson":
            calc_pearson = True
        elif measure == "mutual_info":
            calc_mutual = True
        elif measure == "spearman":
            calc_spearman = True
        elif measure == "max":
            calc_pearson = True
            calc_mutual = True
            calc_spearman = True
            calc_max = True
        else:
            raise ValueError(
                "Unrecognized dependence measure {}. Valid measures are {}".format(
                    measure,
                    CONFIG_DEFAULTS["correlation_metrics"],
                ),
            )

    if calc_pearson:
        calc_order.append("pearson")
    if calc_spearman:
        calc_order.append("spearman")
    if calc_mutual:
        calc_order.append("mutual_info")

    return measures, calc_order, calc_max
