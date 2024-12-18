from statsmodels.formula.api import (
    ols as smf_ols,
    logit as smf_logit,
    probit as smf_probit,
)
from numpy import (
    mean as np_mean,
    number as np_number,
    random as np_random,
    column_stack as np_column_stack,
)
import pandas as pd
from scipy import stats as scipy_stats
from .plots import _plot_actual_fitted
from statsmodels.stats.diagnostic import linear_reset

# numpy required for use in patsy formulae
from numpy import log, exp, floor, ceil, trunc, absolute  # noqa: F401


class LinearModels:
    """Base class for linear models. This class wraps statsmodels' RegressionResults and provides additional methods relevant to linear models."""

    def __init__(self, formula, data, **kwargs):
        self.model = smf_ols(formula, data)
        self.data = data

    def summary(self):
        return self.result.summary(slim=True)

    def cite(self):
        """Returns citations for the source of the model."""

        citation = "Seabold, Skipper, and Josef Perktold. “statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010."
        citation_bibtex_entry = """
        @inproceedings{seabold2010statsmodels,
        title={statsmodels: Econometric and statistical modeling with python},
        author={Seabold, Skipper and Perktold, Josef},
        booktitle={9th Python in Science Conference},
        year={2010},
        }
        """
        print("Cite statsmodels for scientific publications:")
        print(citation)
        print("\nCite statsmodels using bibtex:")
        print(citation_bibtex_entry)
        return

    def prediction_intervals(self, exog=None, alpha=0.05):
        """Returns a table of prediction intervals."""
        predictions = self.result.get_prediction(exog)
        prediction_table = predictions.summary_frame(alpha=alpha)
        prediction_table = prediction_table[["mean", "obs_ci_lower", "obs_ci_upper"]]
        prediction_table = prediction_table.rename(
            columns={
                "mean": "Predicted",
                "obs_ci_lower": "Lower",
                "obs_ci_upper": "Upper",
            }
        )
        return prediction_table

    def standardized_coefficients(self):
        """Returns a table of the standardized coefficients."""
        # standardize data
        df_z = (
            self.data.select_dtypes(include=[np_number])
            .dropna()
            .apply(scipy_stats.zscore)
        )
        result = smf_ols(self.model.formula, data=df_z).fit()
        # drop 'Intercept
        return result.params[1:]

    def elasticities(self):
        """Returns a table of the elasticities."""
        # drop 'Intercept' from rhs
        rhs = self.model.exog_names[1:]
        lhs = self.model.endog_names

        means = self.data[rhs].mean()
        y_mean = self.data[lhs].mean()
        # drop 'Intercept'
        coefs = self.result.params[1:]

        elasticities = coefs * (means / y_mean)

        return round(elasticities, 4)

    def scaled_coefficients(self):
        """Returns a table of both standardized cofficients and elasticities."""
        std_coefs = self.standardized_coefficients()
        elasticities = self.elasticities()
        # drop 'Intercept'
        coefs = self.result.params[1:]
        table = pd.DataFrame(
            dict(coefs=coefs, std_coefs=std_coefs, elasticities=elasticities)
        )
        return table

    def plot_actual_fitted(self):
        """Plots actual values and predicted values with upper and lower prediction intervals for the given linear model."""
        y_id = self.model.endog_names
        y = self.data[y_id]
        X = self.data[self.model.exog_names[1:]]
        Pred_and_PI = self.result.get_prediction(X).summary_frame(alpha=0.05)
        predicted = Pred_and_PI["mean"]
        lower = Pred_and_PI["obs_ci_lower"]
        upper = Pred_and_PI["obs_ci_upper"]
        _plot_actual_fitted(y, y_id, predicted, upper, lower)

    def wald_test(self, hypothesis):
        """Test for linear relationships among multiple coefficients.

        Args:
            hypothesis: The test hypothesis.
        """
        # Statsmodels FutureWarning: The behavior of wald_test will change after 0.14 to returning scalar test statistic values.
        # To get the future behavior now, set scalar to True.
        wald_test = self.result.wald_test(r_matrix=hypothesis, use_f=True, scalar=True)
        print("p-value: ", round(wald_test.pvalue, 4))
        return

    def ramsey_test(self):
        """Model specification test used to test functional form. The Ramsey Test is often called the "Ramsey RESET test" which stands for "Ramsey Regression Equation Specification Error Test."

        Power notation is different in Python's statsmodels and R. Power=2 in Python is equivalent to Power=1 in R.

        Returns:
            : Frame with columns [power, pvalue]
        """
        fit = self.model.fit()
        pvalues = [
            {
                "power": 2,
                "pvalue": linear_reset(fit, use_f=True, power=2).pvalue,
            },
            {
                "power": 3,
                "pvalue": linear_reset(fit, use_f=True, power=3).pvalue,
            },
        ]
        return pd.DataFrame(pvalues)

    def coefficients_and_covariance(self):
        return (self.result.params, self.result.cov_params())

    def coefficients_and_covariance_table(self):
        coefs, cov_matrix = self.coefficients_and_covariance()
        combined = np_column_stack((coefs, cov_matrix))
        rhs = self.model.exog_names  # keep Intercept
        table = pd.DataFrame(combined, columns=["coefs", *rhs])
        table.insert(0, "", rhs)
        return table

    def coefficients_draw(self, size=100000):
        coefs, cov_matrix = self.coefficients_and_covariance()
        rng = np_random.default_rng()
        coef_draws = rng.multivariate_normal(coefs, cov_matrix, size)
        rhs = self.model.exog_names  # keep Intercept
        return pd.DataFrame(coef_draws, columns=rhs)

    def __getattr__(self, name):
        # Delegates any method calls not explicitly defined here to the wrapped object
        return getattr(self.result, name)


class TimeSeriesLinearModel(LinearModels):
    def __init__(self, formula, data, maxlags=2, use_correction=True, **kwargs):
        super().__init__(formula, data, **kwargs)
        self.result = self.model.fit().get_robustcov_results(
            cov_type="HAC", maxlags=maxlags, use_correction=use_correction
        )


class CrossSectionLinearModel(LinearModels):
    def __init__(self, formula, data, **kwargs):
        super().__init__(formula, data, **kwargs)
        self.result = self.model.fit(cov_type="HC1")


class BinaryChoiceModels:
    """Base class for Binary Choice Models. Provides general methods related to binary choice models."""

    def __init__(self, formula, data):
        raise NotImplementedError

    def predict_and_rank(self, exog):
        prospects = exog
        prospects["PredictionNew"] = self.predict(exog)
        prospects["ProspectRank"] = prospects["PredictionNew"].rank()
        return prospects

    def classification_table(self, p_cutoff=None):
        if p_cutoff:
            threshold = p_cutoff
        else:
            threshold = np_mean(self.predict())

        frequency = self.result.pred_table(threshold).flatten().tolist()
        print(frequency)
        table = pd.DataFrame(
            {
                "Summary": ["Correct", "Incorrect", "Incorrect", "Correct"],
                "Actual": [0, 0, 1, 1],
                "Predicted": [0, 1, 0, 1],
                "Frequency": frequency,
            }
        )
        return table

    def marginal_effects(self):
        marginal_effects_at_the_mean = self.result.get_margeff(at="overall")
        return marginal_effects_at_the_mean.summary()

    def wald_testing(self, hypothesis):
        wald_test = self.result.wald_test(r_matrix=hypothesis, scalar=True)
        print("Wald Test Statistic: ", wald_test.statistic)
        print("p-value: ", wald_test.pvalue)
        return

    def __getattr__(self, name):
        # Delegates any method calls not explicitly defined here to the wrapped object
        return getattr(self.result, name)


class LogitModel(BinaryChoiceModels):
    """Logit Model."""

    def __init__(self, formula, data):
        self.model = smf_logit(formula, data)
        self.result = self.model.fit()
        self.data = data


class ProbitModel(BinaryChoiceModels):
    """Probit Models."""

    def __init__(self, formula, data):
        self.model = smf_probit(formula, data)
        self.result = self.model.fit()
        self.data = data


def model(name, formula, data, **kwargs):
    """A factory function for constructing models based on the model's name.

    Args:
        name (str): Name of model to be constructed.
        formula (formula_like): Formula used to construct the model object.
        data (Dataframe): Data to fit the model.

    Raises:
        ValueError: Raises when the name value is not recognized.

    Returns:
        ModelWrapper: A model object.
    """
    match name:
        case "cs":
            return CrossSectionLinearModel(formula, data)
        case "ts":
            return TimeSeriesLinearModel(formula, data)
        case "logit":
            return LogitModel(formula, data)
        case "probit":
            return ProbitModel(formula, data)
        case _:
            msg = f'Model name "{name}" not recognized.'
            print(msg)
            raise ValueError(msg)
