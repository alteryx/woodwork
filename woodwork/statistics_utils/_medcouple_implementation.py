import numpy as np


def _calculate_medcouple_statistic(y):
    """Calculates the medcouple statistic. Based on the implementation by statsmodels and off of the paper
    M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed distributions" Computational Statistics
    & Data Analysis, vol. 52, pp. 5186-5201, August 2008.

    Args:
        y (np.ndarray, pd.Series): Data on which sampling will occur.

    Returns:
        float: Medcouple statistic
    """
    y = np.squeeze(np.asarray(y))
    y = np.sort(y)

    y_length = y.shape[0]
    if y_length % 2 == 0:
        median = (y[y_length // 2 - 1] + y[y_length // 2]) / 2
    else:
        median = y[(y_length - 1) // 2]

    deviations = y - median
    lower = deviations[deviations <= 0.0]
    upper = deviations[deviations >= 0.0]
    upper = upper[:, None]
    standardization = upper - lower
    is_zero = np.logical_and(lower == 0.0, upper == 0.0)
    standardization[is_zero] = np.inf
    spread = upper + lower
    h = spread / standardization
    num_ties = np.sum(lower == 0.0)
    if num_ties:
        replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
        replacements -= 2 * np.triu(replacements)
        replacements = np.fliplr(replacements)
        h[:num_ties, -num_ties:] = replacements
    return np.median(h)
