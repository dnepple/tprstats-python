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
    dist = getattr(stats, distribution)
    params = dist.fit(data)
    log_likelihood = np.sum(np.log(dist.pdf(data, *params)))
    aic = (len(params) - 2 * log_likelihood) / len(data)
    return aic


def list_aic(data):
    all_aic = {}

    distributions = (
        DISTRIBUTIONS if data.min() < 0 else DISTRIBUTIONS + POSITIVE_ONLY_DISTRIBUTIONS
    )

    for distribution in distributions:
        aic = calculate_aic(distribution, data)
        all_aic[distribution] = aic
    return Series(all_aic)


def select_distribution(data):
    all_aic = list_aic(data)
    best_aic = all_aic.idxmin()
    return best_aic
