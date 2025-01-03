import numpy as np
from scipy import stats
import pandas as pd

# data = [1, 2, 3, 4, 5]
# m,s = norm.fit(data)
# log_likelihood = np.sum(np.log(norm.pdf(data,m,s)))

df = pd.read_csv("../../data/Sample_from_Normal_Distribution.csv")
thedata = df["x"]
print(thedata)

distributions = [
    "norm",
    "uniform",
    "t",
    "skewnorm",
    "weibull_min",
    "gamma",
    "lognorm",
    "expon",
]


def calculate_aic(dist, data):
    params = dist.fit(data)
    log_likelihood = np.sum(np.log(dist.pdf(data, *params)))
    aic = 2 * len(params) - 2 * log_likelihood
    return aic


def list_aic(distributions, data):
    for distribution in distributions:
        dist = getattr(stats, distribution)
        aic = calculate_aic(dist, data)
        print(f"Distribution: {distribution} aic is {aic}.")


list_aic(distributions, thedata)
