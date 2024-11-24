from abc import ABC, abstractmethod
import statsmodels.formula.api as smf
from numpy import mean as numpy_mean
import pandas as pd


class ModelWrapper(ABC):
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def result(self):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def formula(self):
        pass

    @abstractmethod
    def data(self):
        pass


class StatsmodelsModelWrapper(ModelWrapper):
    def model(self):
        return self._model

    def result(self):
        return self._result

    def summary(self):
        return self._summary

    def formula(self):
        return self._formula

    def data(self):
        return self._data

    def cite(self):
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


class CrossSectionalLinearModel(StatsmodelsModelWrapper):
    def __init__(self, formula, data):
        super().__init__()
        self._model = smf.ols(formula, data)
        self._result = self._model.fit().get_robustcov_results(cov_type="HC1")
        self._summary = self._result.summary(slim=True)
        self._formula = formula
        self._data = data


class TimeSeriesLinearModel(StatsmodelsModelWrapper):
    def __init__(self, formula, data, maxlags=1):
        super().__init__()
        self._model = smf.ols(formula, data)
        self._result = self._model.fit().get_robustcov_results(
            cov_type="HAC", maxlags=maxlags
        )
        self._summary = self._result.summary(slim=True)
        self._formula = formula
        self._data = data


class LogitModel(StatsmodelsModelWrapper):
    def __init__(self, formula, data):
        super().__init__()
        self._model = smf.logit(formula, data)
        self._result = self._model.fit()
        self._summary = self._result.summary(slim=True)
        self._formula = formula
        self._data = data

    def predict(self, *args):
        return self._model.predict(params=args)

    def classification_table(self, p_cutoff=None, *args):
        mypred = self.predict(params=args)
        if not p_cutoff:
            p_cutoff = numpy_mean(mypred)

        actual = self._data[args]
        predicted = pd.to_numeric(mypred > p_cutoff)
        print(f"p cutoff is {p_cutoff}")
        return pd.merge(actual, predicted)


#     logitClassificationTable <- function(mylogit, myvar, data, p_cutoff = NULL) {
#     mypred <- stats::predict(mylogit, newdata = data, type = "response")
#     if (is.null(p_cutoff)) {
#         p_cutoff <- mean(mypred)
#     }
#     Actual <- data[[myvar]]
#     Predicted <- as.numeric(mypred > p_cutoff)
#     Summary <- c("Correct", "Incorrect", "Incorrect", "Correct")
#     cat("p_cutoff is ", p_cutoff, "\n")
#     data.frame(Summary, table(Actual, Predicted))

# }


class ProbitModel(StatsmodelsModelWrapper):
    def __init__(self, formula, data):
        super().__init__()
        self._model = smf.logit(formula, data)
        self._result = self._model.fit()
        self._summary = self._result.summary(slim=True)
        self._formula = formula
        self._data = data


def model(name, formula, data, **kwargs):
    match name:
        case "cs":
            return CrossSectionalLinearModel(formula, data)
        case "ts":
            if "maxlags" in kwargs:
                return TimeSeriesLinearModel(formula, data, maxlags=kwargs["maxlags"])
            else:
                return TimeSeriesLinearModel(formula, data)
        case _:
            msg = f'Model name "{name}" not recognized.'
            print(msg)
            raise ValueError(msg)
