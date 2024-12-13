from abc import ABC, abstractmethod
import statsmodels.formula.api as smf
from numpy import mean as numpy_mean
from numpy import number as numpy_number
from numpy import column_stack
import pandas as pd
from scipy import stats as scipy_stats
from .plots import _plot_actual_fitted

# numpy required for use in patsy formulae
from numpy import log, exp, floor, ceil, trunc, absolute  # noqa: F401


class _ModelWrapper(ABC):
    """Defines a uniform interface for interacting with model objects. All model objects should, at a minimum, implement this interface."""

    @abstractmethod
    def model(self):
        """Returns a model."""
        pass

    @abstractmethod
    def result(self):
        """Returns the result of fitting the model."""
        pass

    @abstractmethod
    def summary(self):
        """Return summary statistics for the given model."""
        pass

    @abstractmethod
    def formula(self):
        """Returns the model's formula."""
        pass

    @abstractmethod
    def data(self):
        """Returns the dataframe used to fit the model."""
        pass


class _StatsmodelsModelWrapper(_ModelWrapper):
    """Wraps models from the statsmodels package."""

    def model(self):
        return self._model

    def result(self):
        return self._result

    def summary(self):
        return self._summary

    def formula(self):
        return self._formula

    def predict(self, exog=None):
        return self._result.predict(exog)

    def data(self):
        return self._data

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


class _LinearModels(_StatsmodelsModelWrapper):
    """An abstract class defining general methods for linear models."""

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
        result = smf.ols(self._formula, data=df_z).fit()
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

    def coefficients_and_covariance(self):
        return (self._result.params, self._result.cov_params())

    def coefficients_and_covariance_table(self):
        coefs, cov_matrix = self.coefficients_and_covariance()
        combined = column_stack((coefs, cov_matrix))
        rhs = self._model.exog_names  # keep Intercept
        table = pd.DataFrame(combined, columns=["coefs", *rhs])
        table.insert(0, "", rhs)
        return table


class CrossSectionalLinearModel(_LinearModels):
    """A concrete class for cross-sectional linear models."""

    def __init__(self, formula, data):
        super().__init__()
        self._model = smf.ols(formula, data)
        self._result = self._model.fit().get_robustcov_results(cov_type="HC1")
        self._summary = self._result.summary(slim=True)
        self._formula = formula
        self._data = data


class TimeSeriesLinearModel(_LinearModels):
    """A concrete class for time series linear models."""

    def __init__(self, formula, data, maxlags=1):
        super().__init__()
        self._model = smf.ols(formula, data)
        self._result = self._model.fit().get_robustcov_results(
            cov_type="HAC", maxlags=maxlags
        )
        self._summary = self._result.summary(slim=True)
        self._formula = formula
        self._data = data


class _BinaryChoiceModels(_StatsmodelsModelWrapper):
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
            return CrossSectionalLinearModel(formula, data)
        case "ts":
            if "maxlags" in kwargs:
                return TimeSeriesLinearModel(formula, data, maxlags=kwargs["maxlags"])
            else:
                return TimeSeriesLinearModel(formula, data)
        case "logit":
            return LogitModel(formula, data)
        case "probit":
            return ProbitModel(formula, data)
        case _:
            msg = f'Model name "{name}" not recognized.'
            print(msg)
            raise ValueError(msg)
