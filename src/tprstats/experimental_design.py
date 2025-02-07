from numpy import sqrt as np_sqrt
from statsmodels.stats.power import tt_ind_solve_power
from pandas import Series


def pooled_std(s1, s2, n1, n2):
    """Calculates the pooled standard deviation, i.e., the difference in means between two independent samples.

    Args:
        s1 : standard deviation of group 1
        s2 : standard deviation of group 2
        n1 : number of observations from group 1
        n2 : number of observations from group 2

    Returns: Pooled standard deviation
    """
    return np_sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))


def cohens_d(m1, m2, s1, s2, n1, n2):
    """Calculates Cohen's d a.k.a. the standardized effect size.

    Args:
        m1 : mean of group 1
        m2 : mean of group 2
        s1 : standard deviation of group 1
        s2 : standard deviation of group 2
        n1 : number of observations from group 1
        n2 : number of observations from group 2

    Returns: Cohen's d
    """
    pooled_sd = pooled_std(s1, s2, n1, n2)

    d = (m1 - m2) / pooled_sd
    return d


def AB_nobs(diff_means, std1, std2, power, ratio=1, alpha=0.05):
    """Calculates the number of observations needed for an experiment to test a difference in population means.

    Args:
        diff_means : The difference in means.
        std1 : standard deviation of group 1
        std1 : standard deviation of group 1
        power : Desired power
        ratio (float, optional): Ratio of nobs2/nobs1. Defaults to 1.
        alpha (float, optional): The significance level. Defaults to 0.05.

    Returns:
        pandas.Series: The number of observations for an experiment.
    """
    # n1_start and n2_start are starting values passed to the solver
    nobs2_start = 30
    nobs1_start = nobs2_start / ratio
    m1 = diff_means
    m2 = 0
    nobs1 = tt_ind_solve_power(
        effect_size=cohens_d(m1, m2, std1, std2, nobs1_start, nobs2_start),
        nobs1=None,
        alpha=alpha,
        power=power,
        ratio=ratio,
    )
    # ratio as defined by stats.power.tt_ind_solve_power
    nobs2 = nobs1 * ratio
    nobs_total = nobs1 + nobs2
    return Series({"nobs1": nobs1, "nobs2": nobs2, "nobs_total": nobs_total})


def AB_power(diff_means, nobs1, nobs2, std1, std2, alpha=0.05):
    """Calculates the power of an experiment to test a difference in population means.

    Args:
        diff_means : The difference in means
        nobs1 : The number of observations from population 1
        nobs2 : The number of observations from population 2
        std1 : The standard deviation for population 1
        std2 : The standard deviation for population 2
        alpha (float, optional): The significance level. Defaults to 0.05.

    Returns: Power
    """
    # ratio as defined by stats.power.tt_ind_solve_power
    ratio = nobs2 / nobs1
    m1 = diff_means
    m2 = 0
    return tt_ind_solve_power(
        effect_size=cohens_d(m1, m2, std1, std2, nobs1, nobs2),
        nobs1=nobs1,
        alpha=alpha,
        ratio=ratio,
    )
