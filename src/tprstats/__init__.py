__all__ = [
    "model",
    "control_chart",
    "control_chart_binary",
    "plot_3D",
    "hist_CI",
    "calculate_aic",
    "list_aic",
    "select_distribution",
    "test_population_mean",
]
# read version from installed package
from importlib.metadata import version

__version__ = version("tprstats")

from .models import model
from .plots import control_chart, control_chart_binary, plot_3D, hist_CI
from .fit_distribution import (
    calculate_aic,
    list_aic,
    select_distribution,
)
from .hypothesis_test import test_population_mean
