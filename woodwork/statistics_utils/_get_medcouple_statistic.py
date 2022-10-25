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

    n = y.shape[0]
    if n % 2 == 0:
        mf = (y[n // 2 - 1] + y[n // 2]) / 2
    else:
        mf = y[(n - 1) // 2]

    z = y - mf
    lower = z[z <= 0.0]
    upper = z[z >= 0.0]
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


def _get_medcouple(array_, axis=0):
    array_ = np.asarray(array_, dtype=np.double)
    mc = np.apply_along_axis(_calculate_medcouple_statistic, axis, array_)
    if isinstance(mc, np.ndarray) and isinstance(mc.tolist(), float):
        return mc.tolist()
    return mc
