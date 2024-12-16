import statsmodels.formula.api as smf
from numpy import mean as numpy_mean
from numpy import number as numpy_number
from numpy import random as np_random
from numpy import column_stack
import pandas as pd
from scipy import stats as scipy_stats
from .plots import _plot_actual_fitted
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.formula.api import ols

# numpy required for use in patsy formulae
from numpy import log, exp, floor, ceil, trunc, absolute  # noqa: F401


class LinearModelsMixin:
    """A mixin class providing additional methods for linear regression."""

    def prediction_intervals(self, exog=None, alpha=0.05):
        """Returns a table of prediction intervals."""
        predictions = self._result.get_prediction(exog)
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
            self._data.select_dtypes(include=[numpy_number])
            .dropna()
            .apply(scipy_stats.zscore)
        )
        result = smf.ols(self.model.formula, data=df_z).fit()
        # drop 'Intercept
        return result.params[1:]

    def elasticities(self):
        """Returns a table of the elasticities."""
        # drop 'Intercept' from rhs
        rhs = self._model.exog_names[1:]
        lhs = self._model.endog_names

        means = self._data[rhs].mean()
        y_mean = self._data[lhs].mean()
        # drop 'Intercept'
        coefs = self._result.params[1:]

        elasticities = coefs * (means / y_mean)

        return round(elasticities, 4)

    def scaled_coefficients(self):
        """Returns a table of both standardized cofficients and elasticities."""
        std_coefs = self.standardized_coefficients()
        elasticities = self.elasticities()
        # drop 'Intercept'
        coefs = self._result.params[1:]
        table = pd.DataFrame(
            dict(coefs=coefs, std_coefs=std_coefs, elasticities=elasticities)
        )
        return table

    def plot_actual_fitted(self):
        """Plots actual values and predicted values with upper and lower prediction intervals for associated linear model."""
        y_id = self._model.endog_names
        y = self._data[y_id]
        X = self._data[self._model.exog_names[1:]]
        Pred_and_PI = self._result.get_prediction(X).summary_frame(alpha=0.05)
        predicted = Pred_and_PI["mean"]
        lower = Pred_and_PI["obs_ci_lower"]
        upper = Pred_and_PI["obs_ci_upper"]
        _plot_actual_fitted(y, y_id, predicted, upper, lower)

    def wald_testing(self, hypothesis):
        """Test for linear relationships among multiple coefficients.

        Args:
            hypothesis: The test hypothesis.

        Returns:
            : P-value
        """
        # Statsmodels FutureWarning: The behavior of wald_test will change after 0.14 to returning scalar test statistic values.
        # To get the future behavior now, set scalar to True.
        wald_test = self._result.wald_test(r_matrix=hypothesis, use_f=True, scalar=True)
        print("Wald Test Statistic: ", wald_test.statistic)
        print("p-value: ", wald_test.pvalue)
        return

    def ramsey_test(self):
        """Model specification test used to test functional form. The Ramsey Test is often called the "Ramsey RESET test" which stands for "Ramsey Regression Equation Specification Error Test."

        Power nomenclature is different in Python's statsmodels and R. Power=2 in Python is equivalent to Power=1 in R.

        Returns:
            : Frame with columns [power, pvalue]
        """
        fit = self._model.fit()
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
        return (self._result.params, self._result.cov_params())

    def coefficients_and_covariance_table(self):
        coefs, cov_matrix = self.coefficients_and_covariance()
        combined = column_stack((coefs, cov_matrix))
        rhs = self._model.exog_names  # keep Intercept
        table = pd.DataFrame(combined, columns=["coefs", *rhs])
        table.insert(0, "", rhs)
        return table

    def coefficients_draw(self, size=100000):
        coefs, cov_matrix = self.coefficients_and_covariance()
        rng = np_random.default_rng()
        coef_draws = rng.multivariate_normal(coefs, cov_matrix, size)
        rhs = self._model.exog_names  # keep Intercept
        return pd.DataFrame(coef_draws, columns=rhs)


class StatsmodelsLinearModelsWrapper(LinearModelsMixin):
    """Wrapper class for statsmodels RegressionResults."""

    def __init__(self, formula, data, **kwargs):
        self._model = ols(formula, data)
        self._data = data

        # cross_section = {"cov_type": "HC1"}
        # time_series = {"cov_type": "HAC", "maxlags": 1}
        self._result = self._model.fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    def summary(self):
        return self._result.summary(slim=True)

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

    def __getattr__(self, name):
        # Delegates any method calls not explicitly defined here to the wrapped object
        return getattr(self._result, name)


class _BinaryChoiceModels:
    """An abstract class defining general methods for binary choice models."""

    def predict_and_rank(self, exog):
        prospects = exog
        prospects["PredictionNew"] = self.predict(exog)
        prospects["ProspectRank"] = prospects["PredictionNew"].rank()
        return prospects

    def classification_table(self, p_cutoff=None):
        if p_cutoff:
            threshold = p_cutoff
        else:
            threshold = numpy_mean(self.predict())

        frequency = self._result.pred_table(threshold).flatten().tolist()
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
        marginal_effects_at_the_mean = self._result.get_margeff(at="overall")
        return marginal_effects_at_the_mean.summary()

    def wald_test(self, hypothesis):
        wald_test = self._result.wald_test(r_matrix=hypothesis, scalar=True)
        print("Wald Test Statistic: ", wald_test.statistic)
        print("p-value: ", wald_test.pvalue)
        return


class LogitModel(_BinaryChoiceModels):
    """A concrete class for Logit Models."""

    def __init__(self, formula, data):
        super().__init__()
        self._model = smf.logit(formula, data)
        self._result = self._model.fit()
        self._summary = self._result.summary()
        self._formula = formula
        self._data = data


class ProbitModel(_BinaryChoiceModels):
    """A concrete class for Probit Models."""

    def __init__(self, formula, data):
        super().__init__()
        self._model = smf.probit(formula, data)
        self._result = self._model.fit()
        self._summary = self._result.summary()
        self._formula = formula
        self._data = data


def model(name, formula, data, **kwargs):
    """A factory function for constructing models based on the name param.

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
            return StatsmodelsLinearModelsWrapper(formula, data, cov_type="HC1")
        case "ts":
            return StatsmodelsLinearModelsWrapper(
                formula=formula, data=data, cov_type="HAC", cov_kwds={"maglags": 1}
            )
        case "logit":
            return LogitModel(formula, data)
        case "probit":
            return ProbitModel(formula, data)
        case _:
            msg = f'Model name "{name}" not recognized.'
            print(msg)
            raise ValueError(msg)
