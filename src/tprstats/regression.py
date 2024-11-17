import statsmodels.formula.api as smf
import pandas as pd


def lm(formula, data):
    """Fits a linear model to data using ordinary least squares(OLS). Includes robust standard errors.

    Args:
        formula (str or Formula object): Formula specifying the model.
        data (array_like): The data for the model.

    Returns:
        model (Model): The fitted model.
    """
    return smf.ols(formula, data).fit(cov_type="HC1")


def summary(model):
    """Prints summary statistics for the given model.

    Args:
        model (Model): The model.
    """
    print(model.summary(slim=True))
    return
