from scipy.stats import ttest_1samp, sem, t
from numpy import mean as np_mean
from pandas import Series


def test_population_mean(data, popmean, alpha=0.05):
    """
    Test of a null hypothesis about a sample mean.
    Args:
        data : Data
        popmean (float): Population Mean
        alpha (float, optional): Significance Level. Defaults to 0.05.

    Returns:
        pandas.Series: Returns the t-statistic, p_value, confidence interval and sample.
    """
    t_statistic, p_value = ttest_1samp(data, popmean=popmean)
    confidence_level = 1 - alpha
    degrees_freedom = len(data) - 1
    sample_mean = np_mean(data)
    sample_standard_error = sem(data)
    confidence_interval_lower, confidence_interval_upper = t.interval(
        confidence_level, degrees_freedom, sample_mean, sample_standard_error
    )

    result = Series(
        {
            "t_statistic": t_statistic,
            "p_value": p_value,
            "confidence_interval_lower": confidence_interval_lower,
            "confidence_interval_upper": confidence_interval_upper,
            "sample_mean": sample_mean,
        }
    )
    return result
