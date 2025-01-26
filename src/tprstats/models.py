from statsmodels.api import OLS as sm_OLS, Logit as sm_Logit, Probit as sm_Probit
from patsy import dmatrices as design_matrices
from statsmodels.tsa.api import ARIMA as sm_ARIMA
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

class ExogMixin:

    def exog_as_dmatrix(self, exog):
        """Creates a design matrix for out-of-sample data using the model formula that matches the columns keys in the model.s
        """
        # If the exog dataframe does not include a y column, we need to create a y column.
        # We'll delete it later.
        y_id = self.y.columns[0]
        if not y_id in exog.columns:
            exog[y_id] = [0] * len(exog)
        # Adding y values allows us to reuse the model's formula to construct a the exog dmatrix.
        # Here we get the exog dmatrix and discard the y matrix
        _, exog = design_matrices(self.formula, data = exog, return_type="dataframe")
        return exog

class LinearModels(ExogMixin):
    """Base class for linear models. This class wraps statsmodels' RegressionResults and provides additional methods relevant to linear models."""

    def __init__(self, formula, data, **kwargs):
        self.y, self.X = design_matrices(formula, data=data, return_type="dataframe")
        self.model = sm_OLS(self.y, self.X)
        self.data = data
        self.formula = formula

    def summary(self):
        return self.result.summary(slim=True)
    
    def predict(self, exog = None, *args, **kwargs):
        if exog is not None:
            exog = self.exog_as_dmatrix(exog)
        return self.result.predict(exog=exog, *args, **kwargs)

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
        exog = self.exog_as_dmatrix(exog)
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
        """Returns the standardized coefficients.
        A standardized coefficient is the coefficient obtained from a regression in which both the independent variable and the dependent variable are standardized to have mean equal to zero and standard deviation equal to one.

        Returns:
            pandas.Series: Standardized Coefficients
        """
        standardized_coefs = pd.Series()
        for col_name, col_data in self.X.items():
            if col_data.dtype == np_number:
                standardized_coefs[col_name] = (self.params[col_name] * self.X[col_name].std() / self.y.std()).item()
        return standardized_coefs.drop('Intercept')

    def elasticities(self):
        """
        Elasticities evaluated at the means of the variables are calculated for the coefficients of a linear regression model.
        """
        X = self.X.drop('Intercept', axis=1)
        means = X.mean()
        y_mean = self.y.mean().item()
        coefs = self.result.params.drop('Intercept')

        elasticities = coefs * (means / y_mean)

        return round(elasticities, 4)

    def scaled_coefficients(self):
        """
        Returns a pandas.Dataframe containing the standardized cofficients (also known as beta weights) and elasticities.
        A standardized coefficient is the coefficient obtained from a regression in which both the independent variable and the dependent variable are standardized to have mean equal to zero and standard deviation equal to one.
        The function reports three columns which contain the coefficients, the standardized coefficients, and the elasticities evaluated at the means of the variables.
        """
        std_coefs = self.standardized_coefficients()
        elasticities = self.elasticities()
        coefs = self.result.params.drop('Intercept')
        table = pd.DataFrame(
            dict(coefs=coefs, std_coefs=std_coefs, elasticities=elasticities)
        )
        return table

    def plot_actual_fitted(self):
        """Plots actual values and predicted values with upper and lower prediction intervals for the given linear model."""
        y_id = self.y.columns[0]
        y = self.y
        X = self.X
        Pred_and_PI = self.result.get_prediction(X).summary_frame(alpha=0.05)
        predicted = Pred_and_PI["mean"]
        lower = Pred_and_PI["obs_ci_lower"]
        upper = Pred_and_PI["obs_ci_upper"]
        _plot_actual_fitted(y, y_id, predicted, upper, lower)

    def wald_test(self, hypothesis):
        """Test for linear relationships among multiple coefficients.

        Args:
            hypothesis: The test hypothesis specified as a string.
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
            pandas.DataFrame : Frame with columns [power, pvalue]
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
        """Extracts the coefficients and covariance matrix from a statsmodel regression..

        Returns:
            Tuple: Returns a tuple containing (coefficients, covariance_matrix)
        """
        return (self.result.params, self.result.cov_params())

    def coefficients_and_covariance_frame(self):
        """Returns a Pandas DataFrame of the cofficients and covariance matrix.

        Returns:
            pandas.DataFrame: A dataframe containing the cofficients, and covariance matrix.
        """
        coefs, cov_matrix = self.coefficients_and_covariance()
        combined = np_column_stack((coefs, cov_matrix))
        rhs = self.model.exog_names  # keep Intercept
        frame = pd.DataFrame(combined, columns=["coefs", *rhs])
        frame.insert(0, "", rhs)
        return frame

    def coefficients_draw(self, size=1):
        """Given a regression model, this function takes a random draw for the coefficients.

        Args:
            size (int, optional): Number of draws. Defaults to 1.

        Returns:
            pandas.DataFrame : Draw(s) from the coefficients.
        """
        coefs, cov_matrix = self.coefficients_and_covariance()
        rng = np_random.default_rng()
        coef_draws = rng.multivariate_normal(coefs, cov_matrix, size)
        rhs = self.model.exog_names  # keep Intercept
        return pd.DataFrame(coef_draws, columns=rhs)

    def __getattr__(self, name):
        # Delegates any method calls not explicitly defined in this wrapper class to the wrapped object
        return getattr(self.result, name)


class TimeSeriesLinearModel(LinearModels):
    def __init__(self, formula, data, **kwargs):
        kwargs.setdefault("maxlags", 2)
        kwargs.setdefault("use_correction", True)
        super().__init__(formula, data, **kwargs)
        self.result = self.model.fit().get_robustcov_results(cov_type="HAC", **kwargs)


class CrossSectionLinearModel(LinearModels):
    def __init__(self, formula, data, **kwargs):
        super().__init__(formula, data, **kwargs)
        self.result = self.model.fit(cov_type="HC1")


class BinaryChoiceModels(ExogMixin):
    """Base class for Binary Choice Models. Provides general methods related to binary choice models."""

    def __init__(self, formula, data, **kwargs):
        self.y, self.X = design_matrices(formula, data=data, return_type="dataframe")
        self.formula = formula

    def predict(self, exog=None, *args, **kwargs):
        if exog is not None:
            exog = self.exog_as_dmatrix(exog)
        return self.result.predict(exog=exog, *args, **kwargs)

    def predict_and_rank(self, exog):
        """Predict probabilities from a binary choice model and order probabilities from lowest to highest.

        Args:
            exog : New explanatory variables.

        Returns:
            : Predictions by rank.
        """
        if exog is not None:
            exog = self.exog_as_dmatrix(exog)
        prospects = exog
        prospects["PredictionNew"] = self.predict(exog)
        prospects["ProspectRank"] = prospects["PredictionNew"].rank()
        return prospects

    def classification_table(self, p_cutoff=None):
        """Classification table for binary choice models. For a given estimated model, the classification table rovides a summary of the model's predictive accuracy.

        Args:
            p_cutoff (number, optional): Predicted probabilities greater than the p_cutoff are classified as 1, and 0 otherwise. If p_cutoff is unspecified, the p_cutoff is set equal to the mean of the predicted probabilities.

        Returns:
            pandas.DataFrame: The classification table for the associated estimated model.
        """
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
        """
        The average change in probability across all observations in the sample when values of a predictor variable increase by one unit and all other predictor variables are held constant.

        Returns:
            : Marginal effects summary table.
        """
        marginal_effects_at_the_mean = self.result.get_margeff(at="overall")
        return marginal_effects_at_the_mean.summary()

    def wald_test_binary(self, hypothesis):
        """Prints the Wald test stastistic and p-value for the given hypothesis."""
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
        super().__init__(formula, data)
        self.model = sm_Logit(self.y, self.X)
        self.result = self.model.fit()
        self.data = data


class ProbitModel(BinaryChoiceModels):
    """Probit Models."""

    def __init__(self, formula, data):
        super().__init__(formula, data)
        self.model = sm_Probit(self.y, self.X)
        self.result = self.model.fit()
        self.data = data


class ARIMAModel(ExogMixin):
    def __init__(self, formula, data, order=(1, 0, 0), **kwargs):
        # "-1" prevents patsy from adding a constant to the design matrices
        self.formula = formula + "-1"
        self.y, self.X = design_matrices(
            self.formula, data=data, return_type="dataframe"
        )
        self.model = sm_ARIMA(endog=self.y, exog=self.X, order=order, **kwargs)
        self.result = self.model.fit(method="innovations_mle", **kwargs)

    def predict(self, exog, *args, **kwargs):
        if exog is not None:
            exog = self.exog_as_dmatrix(exog)
        return self.result.predict(exog=exog, *args, **kwargs)
    
    def __getattr__(self, name):
        # Delegates any method calls not explicitly defined here to the wrapped object
        return getattr(self.result, name)


def model(name, formula, data, **kwargs):
    """
    Constructs a model for the given model name. Valid names are "cs" for cross-sectional linear model, "ts" for time series linear model, "logit" for logit model, "probit" for probit model,  and "arima" for ARIMA model.

    Args:
        name (str): Name of model to be constructed.
        formula (formula_like): Formula used to construct the model object.
        data (Dataframe): Data to fit the model.

    Raises:
        ValueError: Raises when the name value is not recognized.

    Returns:
        : A model object for the given model name.
    """
    match name:
        case "cs":
            return CrossSectionLinearModel(formula, data, **kwargs)
        case "ts":
            return TimeSeriesLinearModel(formula, data, **kwargs)
        case "logit":
            return LogitModel(formula, data)
        case "probit":
            return ProbitModel(formula, data)
        case "arima":
            return ARIMAModel(formula, data, **kwargs)
        case _:
            msg = f'Model name "{name}" not recognized.'
            print(msg)
            raise ValueError(msg)
