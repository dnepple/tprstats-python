import numpy as np
from scipy import stats
from pandas import Series

DISTRIBUTIONS = [
    "norm",
    "uniform",
    "t",
    "skewnorm",
]

POSITIVE_ONLY_DISTRIBUTIONS = [
    "weibull_min",
    "gamma",
    "lognorm",
    "expon",
]


def calculate_aic(distribution: str, data):
    """Calculate the Akaike Information Criterion (AIC) given a valid scipy.stats distribution namestring.

    Args:
        distribution (str): The name of the distribution. Must be a solution from the package scipy.stats.
        data (pandas.DataFrame): The data.

    Returns:
        float : AIC for the given distribution and data.
    """
    dist = getattr(stats, distribution)
    params = dist.fit(data)
    log_likelihood = np.sum(np.log(dist.pdf(data, *params)))
    aic = (len(params) - 2 * log_likelihood) / len(data)
    return aic


def list_aic(data):
    """
    List AICs for the included distributions.
    Included distributions: normal, uniform, t, skew normal, weibull, gamma, log-normal and exponentional.

    Args:
        data (pandas.DataFrame): The data.

    Returns:
        pandas.Series: The AIC values calculated for the included distributions with respect to the data.
    """
    all_aic = {}

    distributions = (
        DISTRIBUTIONS if data.min() < 0 else DISTRIBUTIONS + POSITIVE_ONLY_DISTRIBUTIONS
    )

    for distribution in distributions:
        aic = calculate_aic(distribution, data)
        all_aic[distribution] = aic
    return Series(all_aic)


def select_distribution(data):
    """
    Fits up to 8 different distributions to a set of data and recommends the best-fitting distribution, where best-fitting is considered to be the distribution with the smallest Akaike Information Criterion (AIC) value.
    Included distributions: normal, uniform, t, skew normal, weibull, gamma, log-normal and exponentional.

    Args:
        data (pandas.DataFrame): The data.

    Returns:
        str : The name of the best-fitting distribution, i.e., the distribution with the lowest AIC.
    """
    all_aic = list_aic(data)
    best_aic = all_aic.idxmin()
    return best_aic
